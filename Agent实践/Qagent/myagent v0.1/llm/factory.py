from config import Settings
from llm.base import BaseLLM
from llm.openai_provider import OpenAIProvider
from llm.deepseek_provider import DeepSeekProvider
from llm.kimi_provider import KimiProvider

# 预设默认值
PROVIDER_DEFAULTS = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
    },
    "kimi": {
        "base_url": "https://api.moonshot.cn/v1",
        "model": "moonshot-v1-8k",
    },
}

def create_llm(cfg: Settings) -> BaseLLM:
    provider = cfg.llm_provider.lower()

    # 获取预设默认值（如果用户没有在配置中覆盖）
    defaults = PROVIDER_DEFAULTS.get(provider, {})
    # 使用配置值，若为空则回退到默认值
    base_url = cfg.llm_base_url or defaults.get("base_url", "https://api.moonshot.cn/v1")
    model = cfg.llm_model or defaults.get("model", "moonshot-v1-8k")

    # 根据 provider 选择实现类
    if provider == "openai":
        cls = OpenAIProvider
    elif provider == "deepseek":
        cls = DeepSeekProvider
    elif provider == "kimi":
        cls = KimiProvider
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

    # 实例化提供者，并注入最终的 base_url 和 model
    instance = cls()
    instance.client.base_url = base_url   # 覆盖默认 base_url
    instance.model = model                # 覆盖模型名
    return instance