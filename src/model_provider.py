from __future__ import annotations

from dataclasses import dataclass


SUPPORTED_PROVIDERS = ("openai", "custom", "gemini", "anthropic", "ollama", "openrouter")

_PROVIDER_ALIASES = {
    "openai": "openai",
    "gpt": "openai",
    "custom": "custom",
    "openai-compatible": "custom",
    "openai_compatible": "custom",
    "compatible": "custom",
    "gemini": "gemini",
    "google": "gemini",
    "google-genai": "gemini",
    "google_genai": "gemini",
    "anthropic": "anthropic",
    "anthorpic": "anthropic",
    "antropic": "anthropic",
    "claude": "anthropic",
    "ollama": "ollama",
    "local": "ollama",
    "openrouter": "openrouter",
    "open-router": "openrouter",
    "open_router": "openrouter",
}

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "custom": "gpt-4o-mini",
    "gemini": "gemini-2.5-flash",
    "anthropic": "claude-sonnet-5-5",
    "ollama": "llama3.1",
    "openrouter": "openai/gpt-4o-mini",
}


@dataclass
class ProviderConfig:
    """Provider configuration shared by the agents.

    Supported providers:
    - openai
    - custom (OpenAI-compatible base URL)
    - gemini
    - anthropic
    - ollama
    - openrouter
    """

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None

    def has_credentials(self) -> bool:
        """Whether a live call is possible with this config."""

        if self.provider == "ollama":
            return bool(self.base_url)
        if self.provider == "custom":
            return bool(self.base_url)
        return bool(self.api_key)


def normalize_provider(value: str) -> str:
    """Map aliases like `anthorpic` -> `anthropic`."""

    key = (value or "").strip().lower()
    if key in _PROVIDER_ALIASES:
        return _PROVIDER_ALIASES[key]
    raise ValueError(f"Unsupported provider {value!r}. Expected one of: {', '.join(SUPPORTED_PROVIDERS)}")


def build_chat_model(config: ProviderConfig):
    """Instantiate the real chat model for the selected provider.

    Provider SDKs are imported lazily so the offline path never needs them.
    """

    provider = normalize_provider(config.provider)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=config.model_name, temperature=config.temperature, api_key=config.api_key)

    if provider == "custom":
        from langchain_openai import ChatOpenAI

        if not config.base_url:
            raise ValueError("Provider `custom` requires a base_url (OpenAI-compatible endpoint).")
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key or "not-needed",
            base_url=config.base_url,
        )

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=config.model_name,
            temperature=config.temperature,
            google_api_key=config.api_key,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=config.model_name, temperature=config.temperature, api_key=config.api_key)

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        kwargs = {"base_url": config.base_url} if config.base_url else {}
        return ChatOllama(model=config.model_name, temperature=config.temperature, **kwargs)

    # openrouter
    from langchain_openrouter import ChatOpenRouter

    return ChatOpenRouter(model=config.model_name, temperature=config.temperature, api_key=config.api_key)
