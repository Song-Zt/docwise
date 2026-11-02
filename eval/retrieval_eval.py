"""
RAG 检索评测：基于 eval/dataset.jsonl，对 Milvus 知识库 (kb_chunks) 做检索质量评估。

评测思路
--------
- 每条测试样本给出用户 query 与标准相关切片 relevant_chunk_ids（chunk_id）。
- 用与生产一致的检索链路召回 Top-K，看标准答案是否命中。
- 评测变体：
    v_embed        仅普通混合检索（稠密0.5+稀疏0.5，带商品名过滤）
    v_embed_nofilter 普通混合检索、不带商品名过滤（用于隔离"过滤条件"的贡献）
    v_hyde_rrf     普通检索 + HyDE 增强检索，再 RRF 融合
    v_full         在 v_hyde_rrf 基础上再过 DashScope 重排 + 断崖截断（= 生产最终上下文）
- 指标：
    Hit@k   标准答案中是否至少有 1 条进入前 k
    Recall@k 前 k 中命中的标准切片数 / 标准切片总数
    MRR@k   第一个命中切片的倒数排名（未命中记 0）

说明
----
- 生产 node_search_embedding / node_search_embedding_hyde 里 hybrid_search 实际按默认 limit=5 截断，
  本脚本对两路召回页取 top-10（保留更多可诊断信息），报告里同时给出 @5 与 @10。
"""
import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pymilvus import MilvusClient

from config.milvus_config import milvus_config
from processor.query_processor.prompt.search_embedding_hyde import HYDE_PROMPT
from tool.logger import logger
from utils.embedding_utils import generate_embeddings
from utils.llm_utils import get_llm_client
from utils.milvus_utils import create_hybrid_search_requests, hybrid_search
from utils.reranker_http_utils import rerank_documents


def load_dataset(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_item_map(client, collection):
    """file_title(item的源文件名) -> 标准化 item_name 集合。用于把测试集 source_file 对齐到库里的商品名。"""
    rows = client.query(collection_name=collection, filter='item_name != ""',
                        output_fields=["item_name", "file_title"], limit=10000)
    mapping = defaultdict(set)
    for r in rows:
        ft = (r.get("file_title") or "").strip()
        it = (r.get("item_name") or "").strip()
        if ft and it:
            mapping[ft].add(it)
    return mapping


def normalize_title(title):
    return (title or "").replace(".pdf", "").strip()


def resolve_item_names(row, item_map, db_item_names):
    """把测试集样本的商品名映射为库内标准化商品名。优先级：source_file -> item_names去空格。"""
    src = normalize_title(row.get("source_file"))
    names = item_map.get(src) or set()
    if not names:
        stripped = [(n or "").replace(" ", "") for n in row.get("item_names", [])]
        names = {n for n in stripped if n in db_item_names}
    return sorted(names)


def search(client, collection, expr, dense, sparse, weights, limit):
    reqs = create_hybrid_search_requests(dense, sparse, expr=expr, limit=limit)
    res = hybrid_search(client, collection, reqs, ranker_weights=weights,
                        limit=limit, output_fields=["chunk_id", "content", "item_name", "title"])
    if not res or not res[0]:
        return []
    return res[0]


def rrf_merge(paths, k=60, max_results=None):
    scores, data = {}, {}
    for path in paths:
        for rank, doc in enumerate(path, start=1):
            cid = doc["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            data.setdefault(cid, doc)
    ordered = sorted(data.values(), key=lambda d: scores[d["chunk_id"]], reverse=True)
    return ordered[:max_results] if max_results else ordered


def hit_at_k(retrieved, relevant, k):
    return 1.0 if set(relevant) & set(retrieved[:k]) else 0.0


def recall_at_k(retrieved, relevant, k):
    relevant = set(relevant)
    if not relevant:
        return 1.0
    return len(relevant & set(retrieved[:k])) / len(relevant)


def mrr_at_k(retrieved, relevant, k):
    relevant = set(relevant)
    for i, cid in enumerate(retrieved[:k]):
        if cid in relevant:
            return 1.0 / (i + 1)
    return 0.0


def eval_one(client, emb, row, item_names, do_hyde, do_rerank, topk):
    query = row["query"]
    relevant = set(int(c) for c in row["relevant_chunk_ids"])
    expr = f"item_name in {[n for n in item_names]}" if item_names else None

    embed_docs = search(client, milvus_config.chunks_collection, expr,
                        emb["dense"][0], emb["sparse"][0], (0.5, 0.5), topk)
    nofilter_docs = search(client, milvus_config.chunks_collection, None,
                           emb["dense"][0], emb["sparse"][0], (0.5, 0.5), topk)

    detail = {
        "query_id": row["query_id"], "query": query,
        "item_names_std": item_names,
        "relevant_ids": sorted(relevant),
        "categories": [row.get("category", "")],
    }

    hyde_docs = []
    if do_hyde:
        try:
            hyde_doc = get_llm_client().invoke(HYDE_PROMPT.format(rewritten_query=query)).content
            combined = query + " " + hyde_doc
            he = generate_embeddings([combined])
            hyde_docs = search(client, milvus_config.chunks_collection, expr,
                               he["dense"][0], he["sparse"][0], (0.8, 0.2), topk)
            detail["hyde_doc"] = hyde_doc
        except Exception as e:
            logger.error(f"[{row['query_id']}] HyDE 失败: {e}")

    rrf_docs = rrf_merge([embed_docs, hyde_docs], max_results=topk)

    final_docs = rrf_docs
    if do_rerank and rrf_docs:
        try:
            scores = rerank_documents(query, [d["content"] for d in rrf_docs])
            scored = sorted(zip(rrf_docs, scores), key=lambda x: x[1], reverse=True)
            reranked = [{**d, "score": s} for d, s in scored]
            final_docs = cliff_cutoff(reranked)
        except Exception as e:
            logger.error(f"[{row['query_id']}] rerank 失败: {e}")
            final_docs = rrf_docs

    detail["embed_ids"] = [d["chunk_id"] for d in embed_docs]
    detail["nofilter_ids"] = [d["chunk_id"] for d in nofilter_docs]
    detail["rrf_ids"] = [d["chunk_id"] for d in rrf_docs]
    detail["final_ids"] = [d["chunk_id"] for d in final_docs]
    return detail


def cliff_cutoff(ranked_docs, max_topk=10, min_topk=3, abs_gap=0.5, rel_gap=0.25):
    if not ranked_docs:
        return []
    upper = min(max_topk, len(ranked_docs))
    lower = min(min_topk, upper)
    cutoff = upper
    for i in range(lower - 1, upper - 1):
        cur, nxt = ranked_docs[i].get("score"), ranked_docs[i + 1].get("score")
        if cur is None or nxt is None:
            continue
        gap = cur - nxt
        if gap >= abs_gap or gap / (abs(cur) + 1e-6) >= rel_gap:
            cutoff = i + 1
            break
    return ranked_docs[:cutoff]


KS = [1, 3, 5, 10]
FINAL_KS = [1, 3, 5]  # 生产断崖截断后的最终上下文为 3~10 条，@10 不设

def aggregate(id_lists, relevant_lists, ks):
    agg = {}
    n = len(id_lists)
    for k in ks:
        if all(len(r) >= k for r in id_lists):
            agg[f"hit@{k}"] = sum(hit_at_k(r, rel, k) for r, rel in zip(id_lists, relevant_lists)) / n
            agg[f"recall@{k}"] = sum(recall_at_k(r, rel, k) for r, rel in zip(id_lists, relevant_lists)) / n
            agg[f"mrr@{k}"] = sum(mrr_at_k(r, rel, k) for r, rel in zip(id_lists, relevant_lists)) / n
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=os.path.join(os.path.dirname(__file__), "dataset.jsonl"))
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--no-hyde", action="store_true")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "results"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = load_dataset(args.dataset)
    logger.info(f"加载测试集 {len(rows)} 条")

    client = MilvusClient(uri=milvus_config.milvus_url)
    item_map = build_item_map(client, milvus_config.chunks_collection)
    db_item_names = set(item_map.keys()) | {it for s in item_map.values() for it in s}

    t0 = time.time()
    logger.info("加载 BGE-M3 embedding 模型（首次约 30~60s）...")
    # 先暖一次模型，避免逐条触发加载锁
    generate_embeddings(["预热模型句子"])

    details = []
    for idx, row in enumerate(rows, 1):
        item_names = resolve_item_names(row, item_map, db_item_names)
        emb = generate_embeddings([row["query"]])
        logger.info(f"[{idx}/{len(rows)}] {row['query_id']} | items={item_names}")
        d = eval_one(client, emb, row, item_names,
                     do_hyde=not args.no_hyde, do_rerank=not args.no_rerank, topk=args.topk)
        details.append(d)

    # 汇总
    relevant_lists = [d["relevant_ids"] for d in details]
    stats = {
        "embed": [d["embed_ids"] for d in details],
        "nofilter": [d["nofilter_ids"] for d in details],
        "rrf": [d["rrf_ids"] for d in details],
        "final": [d["final_ids"] for d in details],
    }
    result = {
        "summary": {
            "embed": aggregate(stats["embed"], relevant_lists, KS),
            "embed_nofilter": aggregate(stats["nofilter"], relevant_lists, KS),
            "embed+hyde+rrf": aggregate(stats["rrf"], relevant_lists, KS),
            "full(含rerank)": aggregate(stats["final"], relevant_lists, FINAL_KS),
        },
        "per_query": details,
    }

    # 输出明细用于诊断
    with open(os.path.join(args.out_dir, "retrieval_detail.jsonl"), "w", encoding="utf-8") as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # 打印汇总
    lines = [f"检索评测结果（{len(details)} 条查询，topk={args.topk}）", "=" * 70]
    for name, agg in result["summary"].items():
        lines.append(f"\n### {name}")
        for k in sorted({int(x.split("@")[1]) for x in agg}):
            h = agg.get(f"hit@{k}", 0); r = agg.get(f"recall@{k}", 0); m = agg.get(f"mrr@{k}", 0)
            lines.append(f"  k={k:<3} Hit@{k}={h:.3f}  Recall@{k}={r:.3f}  MRR@{k}={m:.3f}")

    # 命中最差样本
    lines.append("\n### 各类别的 Hit@5（embed+hyde+rrf）")
    by_cat = defaultdict(list)
    for d in details:
        by_cat[d["categories"][0]].append(d)
    for cat, subs in sorted(by_cat.items()):
        h5 = sum(hit_at_k(s["rrf_ids"], set(s["relevant_ids"]), 5) for s in subs) / len(subs)
        lines.append(f"  {cat:<6} n={len(subs):<3} Hit@5={h5:.3f}")

    lines.append("\n### 检索失败诊断（rrf top5 未命中标准答案）")
    for d in details:
        if not hit_at_k(d["rrf_ids"], set(d["relevant_ids"]), 5):
            lines.append(f"  [{d['query_id']}] {d['query'][:40]} | std_items={d['item_names_std']} | rrf_ids={d['rrf_ids'][:5]}")

    report = "\n".join(lines)
    print(report)
    with open(os.path.join(args.out_dir, "retrieval_report.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")
    logger.info(f"耗时 {time.time() - t0:.1f}s，明细与报告已写入 {args.out_dir}")


if __name__ == "__main__":
    main()