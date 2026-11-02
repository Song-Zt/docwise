import os
from dataclasses import dataclass
from dotenv import load_dotenv
load_dotenv()
@dataclass
class MineruConfig:
    mineru_base_url: str
    mineru_api_key: str

mineru_config = MineruConfig(
    mineru_base_url = os.getenv("MINERU_BASE_URL", ""),
    mineru_api_key = os.getenv("MINERU_API_KEY", "")
)
