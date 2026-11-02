import os
from dataclasses import dataclass
from dotenv import load_dotenv
load_dotenv()
@dataclass
class EmbeddingConfig:
    bge_m3_path: str
    bge_m3: str
    bge_device: str
    beg_fp16: bool

embedding_config = EmbeddingConfig(
    bge_m3_path=os.getenv("BGE_M3_PATH", ""),
    bge_m3=os.getenv("BGE_M3", ""),
    bge_device=os.getenv("BGE_DEVICE", ""),
    beg_fp16=os.getenv("BGE_FP16") in ("1", "Ture", "true", 1)
)