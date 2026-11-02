import os
from dataclasses import dataclass
from dotenv import load_dotenv
load_dotenv()
@dataclass
class LLMConfig:
    openai_base_url: str
    openai_api_key: str
    llm_default_model: str
    llm_default_temperature: float
    vl_model: str
lm_config = LLMConfig(
    openai_base_url = os.getenv("DEEPSEEK_BASE_URL", ""),
    openai_api_key = os.getenv("DEEPSEEK_API_KEY", ""),
    llm_default_model = os.getenv("DEEPSEEK_MODEL", ""),
    llm_default_temperature= float(os.getenv("LLM_DEFAULT_TEMPERATURE", "")),
    vl_model = os.getenv("VL_MODEL", "")
)