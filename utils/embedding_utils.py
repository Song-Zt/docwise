import threading

from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from config.embedding_config import EmbeddingConfig, embedding_config

_bge_m3_ef = None
# 模型初始化锁：导入流程的多文件上传会起多个后台任务线程，若同时执行
# transformers 的 from_pretrained，会残留 meta tensor，随后 .to(device) 报
# "Cannot copy out of meta tensor; no data!"，因此必须串行化首次加载。
_bge_m3_init_lock = threading.Lock()
# 推理锁：保证共享模型不会被多个线程同时执行 encode（首次 encode 内部会调用 .to(device)）
_bge_m3_encode_lock = threading.Lock()

def get_bge_m3_ef():
    """
    获取BGE-M3模型单例对象，自动加载环境变量配置
    :return: 初始化完成的BGEM3EmbeddingFunction实例
    """
    global _bge_m3_ef
    if _bge_m3_ef is not None:
        return _bge_m3_ef

    # 双重检查：并发进入时只允许一个线程真正加载模型
    with _bge_m3_init_lock:
        if _bge_m3_ef is None:
            _bge_m3_ef = BGEM3EmbeddingFunction(
                model_name=embedding_config.bge_m3_path,
                device=embedding_config.bge_device,
                use_fp16=embedding_config.beg_fp16
            )

    return _bge_m3_ef



def generate_embeddings(texts):
    """
    为文本生成向量嵌入
    :param texts: 要生成嵌入的文本列表
    :return: 包含dense和sparse向量的字典
    """
    model = get_bge_m3_ef()
    with _bge_m3_encode_lock:
        embeddings = model.encode_documents(texts)
    processed_sparse = []
    for i in range(len(texts)):
        sparse_indices = embeddings["sparse"].indices[embeddings["sparse"].indptr[i]:embeddings["sparse"].indptr[i+1]].tolist()
        sparse_data = embeddings["sparse"].data[embeddings["sparse"].indptr[i]:embeddings["sparse"].indptr[i+1]].tolist()
        sparse_dict = {k: v for k, v in zip(sparse_indices,sparse_data)}
        processed_sparse.append(sparse_dict)

    return {
        "dense": [emb.tolist() for emb in embeddings["dense"]],
        "sparse": processed_sparse
    }