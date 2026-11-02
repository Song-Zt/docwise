import os
from dataclasses import dataclass
from dotenv import load_dotenv

from utils.embedding_utils import get_bge_m3_ef

load_dotenv()
@dataclass
class MilvusConfig:
    milvus_url: str
    chunks_collection: str
    item_name_collection: str

milvus_config = MilvusConfig(
    milvus_url=os.getenv("MILVUS_URL", ""),
    chunks_collection=os.getenv("CHUNKS_COLLECTION", ""),
    item_name_collection=os.getenv("ITEM_NAME_COLLECTION", "")
)


