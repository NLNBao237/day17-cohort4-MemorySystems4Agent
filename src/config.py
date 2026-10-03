from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import DEFAULT_MODELS, ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    """Shared configuration for the lab.

    - paths for the repo root, dataset directory, and state directory
    - compact-memory settings (threshold + number of messages to keep)
    - provider settings for the main model and the judge model
    - guardrail for persistent memory (confidence threshold before writing `User.md`)
    """

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig
    profile_confidence_threshold: float = 0.6


# provider -> (api key env var, base url env var, default base url)
_PROVIDER_ENV = {
    "openai": ("OPENAI_API_KEY", None, None),
    "custom": ("CUSTOM_API_KEY", "CUSTOM_BASE_URL", None),
    "gemini": ("GEMINI_API_KEY", None, None),
    "anthropic": ("ANTHROPIC_API_KEY", None, None),
    "ollama": (None, "OLLAMA_BASE_URL", "http://localhost:11434"),
    "openrouter": ("OPENROUTER_API_KEY", None, None),
}


def _provider_from_env(prefix: str, fallback: ProviderConfig | None = None) -> ProviderConfig:
    raw_provider = os.getenv(f"{prefix}_PROVIDER")
    if raw_provider is None and fallback is not None:
        provider = fallback.provider
    else:
        provider = normalize_provider(raw_provider or "openai")

    key_env, url_env, default_url = _PROVIDER_ENV[provider]
    same_provider = fallback is not None and fallback.provider == provider
    model_name = os.getenv(f"{prefix}_MODEL") or (fallback.model_name if same_provider else DEFAULT_MODELS[provider])
    temperature = float(os.getenv(f"{prefix}_TEMPERATURE", "0"))
    api_key = os.getenv(key_env) if key_env else None
    if provider == "gemini" and not api_key:
        api_key = os.getenv("GOOGLE_API_KEY")
    base_url = (os.getenv(url_env) if url_env else None) or default_url

    return ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        api_key=api_key or None,
        base_url=base_url,
    )


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load environment variables and return a LabConfig.

    Environment knobs (all optional, offline mode needs none of them):
    - LLM_PROVIDER / LLM_MODEL / LLM_TEMPERATURE
    - JUDGE_PROVIDER / JUDGE_MODEL / JUDGE_TEMPERATURE (defaults to the main model)
    - OPENAI_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY, OPENROUTER_API_KEY
    - OLLAMA_BASE_URL, CUSTOM_BASE_URL / CUSTOM_API_KEY
    - COMPACT_THRESHOLD_TOKENS / COMPACT_KEEP_MESSAGES
    - PROFILE_CONFIDENCE_THRESHOLD
    """

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()

    try:
        from dotenv import load_dotenv

        load_dotenv(root / ".env")
    except ImportError:
        pass

    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    model = _provider_from_env("LLM")
    judge_model = _provider_from_env("JUDGE", fallback=model)

    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=int(os.getenv("COMPACT_THRESHOLD_TOKENS", "800")),
        compact_keep_messages=int(os.getenv("COMPACT_KEEP_MESSAGES", "4")),
        model=model,
        judge_model=judge_model,
        profile_confidence_threshold=float(os.getenv("PROFILE_CONFIDENCE_THRESHOLD", "0.6")),
    )
