# 设备手册 RAG 检索问答系统

面向中文设备/产品手册的**检索增强问答（RAG）**知识库系统。上传 PDF / Markdown 设备手册后，系统自动完成解析、切片、向量化入库，用户即可对设备操作、故障处理、规格参数等内容进行自然语言问答。

## 核心特性

- **文档导入一条龙**：MinerU 云端解析 PDF → 对齐 Markdown 标题切片（含超长文档自动拆分）
- **多模态支持**：MD 内图片经视觉模型理解，与正文一并入库、可随答案展示
- **混合向量检索**：BGE-M3 稠密 + 稀疏双向量，Milvus 加权融合检索
- **多路召回 + 精排**：普通向量检索、HyDE 假设文档检索、联网搜索三路召回，RRF 融合后经 Cross-Encoder（DashScope TextReRank）重排
- **商品名智能确认**：利用向量对齐自动识别用户问题中的产品型号，先确认主体再检索，过滤无关内容
- **流式对话**：SSE 流式输出、多轮会话历史（MongoDB）、检索阶段进度实时推送
- **检索评测体系**：内置带标准答案的评测集与评测脚本，量化 Hit@k / Recall@k / MRR 指标

## 系统架构

### 导入流水线（PDF/MD → 向量库）

```
用户上传文件 (PDF / MD)
   │
   ▼
NodeEntry              按扩展名分发（PDF 或 MD）
   ▼
NodePDFToMD            MinerU API 解析 PDF→Markdown（空密码解密、>200 页自动拆分合并）
   ▼
NodeMDImg              多模态理解 MD 内图片，图片上传 MinIO（VL 模型）
   ▼
NodeDocumentSplit      按 Markdown 标题初切 → 长切短合 → 语义 Chunk
   ▼
NodeItemNameRecognition LLM 识别商品名，商品名向量入库 kb_item_names
   ▼
NodeBgeEmbedding       BGE-M3 生成稠密(dense)+稀疏(sparse) 双向量
   ▼
NodeImportMilvus       幂等入库 kb_chunks（同文件重传自动覆盖，自动建表/索引）
```

### 问答流水线（问题 → 答案）

```
用户问题
   │
   ▼
NodeItemNameConfirm     LLM 抽取商品名 → 向量对齐 kb_item_names → 确认主体（歧义则反问）
   │
   ├─► NodeSearchEmbedding       BGE-M3 混合检索（稠密0.5+稀疏0.5，按 item_name 过滤）
   ├─► NodeSearchEmbeddingHyde   HyDE：LLM 生成假设文档后混合检索（稠密0.8+稀疏0.2）
   └─► NodeWebSearchMcp          联网搜索（MCP，可选，失败不中断）
   │
   ▼
NodeRrf                 多路结果 Reciprocal Rank Fusion 融合排序
   ▼
NodeRerank              DashScope TextReRank 精排 + “断崖”动态截断（保留 3~10 条）
   ▼
NodeAnswerOutput        组装上下文(≤12000 字符) → LLM 流式生成 → 写 MongoDB → SSE 推送
```

## 技术栈

| 类别 | 技术 |
|---|---|
| 语言/环境 | Python ≥ 3.12，uv 包管理 |
| 向量模型 | BGE-M3（稠密 + 稀疏双向量） |
| 向量数据库 | Milvus（WeightedRanker 稠密/稀疏加权融合） |
| 工作流编排 | LangGraph |
| 检索增强 | HyDE、RRF 融合、DashScope TextReRank 重排 |
| LLM | DeepSeek（问答生成）、视觉模型（图片理解、商品名识别） |
| 文档解析 | MinerU（PDF→Markdown） |
| Web 服务 | FastAPI + uvicorn，SSE 流式 |
| 存储 | MinIO（原文件/图片）、MongoDB（会话历史） |

## 目录结构

```
knowledge_base/
├── config/            # 各组件配置（Milvus/Embedding/LLM/MinerU/MinIO/重排/MCP）
├── processor/
│   ├── import_processor/   # 导入流水线（LangGraph 节点化）
│   └── query_processor/    # 问答流水线（LangGraph 节点化 + Prompt）
├── utils/             # embedding/milvus/minio/mongo/rerank/sse/task 等工具
├── web/
│   ├── api/           # FastAPI 服务：query_service(8001) / import_service(8000)
│   └── page/          # 聊天页 / 导入页
├── eval/              # RAG 检索评测（dataset.jsonl + retrieval_eval.py + results/）
├── tests/             # 单元测试（pytest）
├── tool/              # 日志等通用工具
├── pyproject.toml
└── .env               # 环境变量（密钥请勿提交）
```

## 快速开始

### 1. 安装依赖

```bash
uv sync
```

要求 Python ≥ 3.12。首次运行 BGE-M3 模型需按配置加载权重（CPU 约需 1 分钟预热）。

### 2. 配置环境变量

在项目根目录 `.env` 中完成配置（密钥**不要**提交到仓库）。见下方[配置项说明](#配置项说明)。

### 3. 启动服务

```bash
# 查询服务（对话问答）：http://127.0.0.1:8001
uv run python web/api/query_service.py

# 导入服务（上传文档）：http://127.0.0.1:8000
uv run python web/api/import_service.py
```

浏览器打开 [`http://127.0.0.1:8001/chat.html`](http://127.0.0.1:8001/chat.html) 使用聊天页，[`http://127.0.0.1:8000/import.html`](http://127.0.0.1:8000/import.html) 上传文档。

### 4. 主要 API

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/query` | 发起问答（`query` / `session_id` / `is_stream`） |
| GET | `/stream/{session_id}` | SSE 流式接收阶段进度与答案 |
| GET | `/health` | 健康检查 |
| POST | `/upload` | 上传 PDF/MD（多文件），触发导入流水线 |
| GET | `/status/{task_id}` | 轮询导入任务进度与状态 |
| GET | `/swagger` | FastAPI 自带接口文档 |

### 5. 运行测试与评测

```bash
uv run pytest                                   # 单元测试
uv run python eval/retrieval_eval.py            # RAG 检索评测（含 HyDE + 重排）
uv run python eval/retrieval_eval.py --no-rerank # 仅评测召回（embed+hyde+rrf）
uv run python eval/retrieval_eval.py --no-hyde   # 仅评测普通混合检索
```

评测结果输出到 `eval/results/`（汇总报告 + 逐条明细），指标包括 Hit@k、Recall@k、MRR@k，支持按链路拆分对比，用于定位检索瓶颈。

## 配置项说明

`.env` 中主要变量（值请按实际环境填写）：

| 分组 | 变量 | 说明 |
|---|---|---|
| LLM | `DEEPSEEK_API_KEY` `DEEPSEEK_BASE_URL` `DEEPSEEK_MODEL` | 答案生成 LLM（DeepSeek） |
| | `OPENAI_API_KEY` `OPENAI_API_BASE` `LLM_DEFAULT_MODEL` `LLM_DEFAULT_TEMPERATURE` | 通用 LLM（OpenAI 兼容格式） |
| | `VL_MODEL` `ITEM_MODEL` | 视觉模型 / 商品名识别模型 |
| 文档解析 | `MINERU_API_KEY` `MINERU_BASE_URL` | MinerU PDF→Markdown API |
| | `MINERU_MODEL_SOURCE` `MODELSCOPE_CACHE` `HF_HOME` `MD_ROOT_DIR`* | 模型来源与缓存、临时文件根目录 |
| 向量模型 | `BGE_M3_PATH` `BGE_DEVICE` `BGE_FP16` `EMBEDDING_DIM` `EMBEDDING_MODEL` | BGE-M3（稠密+稀疏） |
| Milvus | `MILVUS_URL` `CHUNKS_COLLECTION` `ITEM_NAME_COLLECTION` | 向量库连接与集合名（默认 `kb_chunks` / `kb_item_names`） |
| | `MILVUS_METRIC_TYPE` `MILVUS_MIN_COSINE_SCORE` | 检索度量与最低相似度阈值 |
| 重排 | `TEXT_RERANK_MODEL` `TEXT_RERANK_INSTRUCT` | DashScope TextReRank 精排模型 |
| 联网搜索 | `MCP_DASHSCOPE_BASE_URL` `MCP_API_KEY` `TAVILY_API_KEY` | 搜索 MCP 服务（Tavily 等） |
| MongoDB | `MONGO_URL` `MONGO_DB_NAME` | 会话历史存储 |
| MinIO | `MINIO_ENDPOINT` `MINIO_ACCESS_KEY` `MINIO_SECRET_KEY` `MINIO_BUCKET_NAME` `MINIO_IMG_DIR` | 对象存储（原文件、图片） |
| 其他 | `DATA_BASED_ROOT_DIR` | 上传文件本地根目录 |

> \* 部分变量名以 `.env` 实际内容为准；`BGE_FP16` 按 `1/true` 启用。

## 评测与调优

- 检索评测基于 `eval/dataset.jsonl`（45 条带标准相关切片的查询），以标准 chunk 命中情况评分。
- **已知注意事项**：评测集里的商品名与库内标准化商品名存在差异（空格等细节），评测脚本通过源文件名自动对齐，属预期行为。
- **调优方向**：当前配置下重排环节对 top-1 精度略带回退（将正确片段降权），可尝试改用改写后问题参与重排、更换重排模型，或比较去掉重排直接输出 RRF 结果。

## 说明

- 本系统面向内部/私有设备手册场景；联网搜索为可选增强，未配置或调用失败时自动降级为纯本地检索。
- Milvus、MongoDB、MinIO 需预先部署并在 `.env` 中指向可用地址。