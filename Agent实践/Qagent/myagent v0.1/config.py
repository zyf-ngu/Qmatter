from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    llm_provider: str = "openai"
    llm_api_key: str
    llm_model: str = "gpt-4o-mini"
    llm_base_url: str = "https://api.openai.com/v1"

    workspace_dir: str = "./workspace"
    memory_file: str = "MEMORY.md"
    history_file: str = "HISTORY.md"

    feishu_app_id: str = ""
    feishu_app_secret: str = ""
    feishu_verification_token: str = ""
    feishu_encrypt_key: str = ""

    class Config:
        env_file = str(Path(__file__).parent / ".env")
        env_file_encoding = "utf-8"


settings = Settings()