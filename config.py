"""Central configuration loaded from environment variables / `.env`.

Every tunable value lives here so experiments are reproducible: the active
settings are serialised into each experiment's ``config.json``.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env", override=False)

# Provider -> environment variable holding its API key (None = no key needed)
PROVIDER_KEY_ENV: dict[str, str | None] = {
    "gemini": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "groq": "GROQ_API_KEY",
    "ollama": None,
    "mock": None,
}

DEFAULT_MODELS: dict[str, str] = {
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5",
    "groq": "llama-3.3-70b-versatile",
    "ollama": "llama3.1",
    "mock": "mock-offline-v1",
}

MOCK_MODEL_NAME = DEFAULT_MODELS["mock"]


def _env_str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env_str(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env_str(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    return _env_str(name, str(default)).lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Paths:
    root: Path = PROJECT_ROOT
    data_raw: Path = PROJECT_ROOT / "data" / "raw"
    data_processed: Path = PROJECT_ROOT / "data" / "processed"
    annotations: Path = PROJECT_ROOT / "data" / "annotations"
    outputs: Path = PROJECT_ROOT / "outputs"
    reports: Path = PROJECT_ROOT / "reports"

    @property
    def reviews_csv(self) -> Path:
        return self.data_processed / "reviews.csv"

    @property
    def annotation_sample_csv(self) -> Path:
        return self.annotations / "annotation_sample.csv"

    @property
    def annotations_json(self) -> Path:
        return self.annotations / "annotations.json"

    def mode_dir(self, is_mock: bool) -> Path:
        """Live and mock artefacts are physically separated so they cannot mix."""
        return self.outputs / ("mock" if is_mock else "live")

    @property
    def activity_log(self) -> Path:
        return self.outputs / "activity_log.jsonl"


@dataclass(frozen=True)
class Settings:
    llm_provider: str = "gemini"
    llm_model: str = DEFAULT_MODELS["gemini"]
    llm_temperature: float = 0.0
    llm_max_output_tokens: int = 2048
    llm_timeout_seconds: int = 60
    llm_max_retries: int = 3
    llm_requests_per_minute: int = 10
    judge_provider: str = ""
    judge_model: str = ""
    max_repair_attempts: int = 2
    strict_evidence_grounding: bool = True
    mock_malformed_rate: int = 15
    dataset_category: str = "Appliances"
    dataset_num_products: int = 15
    dataset_reviews_per_product: int = 100
    annotation_sample_size: int = 300
    random_seed: int = 42
    paths: Paths = field(default_factory=Paths)

    # ------------------------------------------------------------------ helpers
    @property
    def is_mock(self) -> bool:
        return self.llm_provider == "mock"

    def api_key_env(self, provider: str | None = None) -> str | None:
        return PROVIDER_KEY_ENV.get(provider or self.llm_provider)

    def has_api_key(self, provider: str | None = None) -> bool:
        env_name = self.api_key_env(provider)
        return env_name is None or bool(_env_str(env_name))

    def live_llm_available(self, provider: str | None = None) -> bool:
        """True when a real (non-mock) LLM can be called with current config."""
        provider = provider or self.llm_provider
        return provider != "mock" and provider in PROVIDER_KEY_ENV and self.has_api_key(provider)

    def live_unavailable_reason(self) -> str:
        if self.llm_provider == "mock":
            return "LLM_PROVIDER is set to 'mock' (offline development mode)."
        if self.llm_provider not in PROVIDER_KEY_ENV:
            return f"Unknown LLM_PROVIDER '{self.llm_provider}'. Use one of {sorted(PROVIDER_KEY_ENV)}."
        env_name = self.api_key_env()
        return (
            f"No API key found for provider '{self.llm_provider}'. "
            f"Add {env_name}=... to your .env file (see .env.example) to enable live LLM features."
        )

    @property
    def effective_judge_provider(self) -> str:
        return self.judge_provider or self.llm_provider

    @property
    def effective_judge_model(self) -> str:
        if self.judge_model:
            return self.judge_model
        if self.judge_provider and self.judge_provider != self.llm_provider:
            return DEFAULT_MODELS.get(self.judge_provider, self.llm_model)
        return self.llm_model

    def public_dict(self) -> dict:
        """Settings safe to write to disk (contains no secrets)."""
        data = asdict(self)
        data["paths"] = {k: str(v) for k, v in asdict(self.paths).items()}
        return data


def load_settings(**overrides) -> Settings:
    provider = _env_str("LLM_PROVIDER", "gemini").lower()
    settings = Settings(
        llm_provider=provider,
        llm_model=_env_str("LLM_MODEL", DEFAULT_MODELS.get(provider, "")),
        llm_temperature=_env_float("LLM_TEMPERATURE", 0.0),
        llm_max_output_tokens=_env_int("LLM_MAX_OUTPUT_TOKENS", 2048),
        llm_timeout_seconds=_env_int("LLM_TIMEOUT_SECONDS", 60),
        llm_max_retries=_env_int("LLM_MAX_RETRIES", 3),
        llm_requests_per_minute=_env_int("LLM_REQUESTS_PER_MINUTE", 10),
        judge_provider=_env_str("JUDGE_PROVIDER").lower(),
        judge_model=_env_str("JUDGE_MODEL"),
        max_repair_attempts=_env_int("MAX_REPAIR_ATTEMPTS", 2),
        strict_evidence_grounding=_env_bool("STRICT_EVIDENCE_GROUNDING", True),
        mock_malformed_rate=_env_int("MOCK_MALFORMED_RATE", 15),
        dataset_category=_env_str("DATASET_CATEGORY", "Appliances"),
        dataset_num_products=_env_int("DATASET_NUM_PRODUCTS", 15),
        dataset_reviews_per_product=_env_int("DATASET_REVIEWS_PER_PRODUCT", 100),
        annotation_sample_size=_env_int("ANNOTATION_SAMPLE_SIZE", 300),
        random_seed=_env_int("RANDOM_SEED", 42),
    )
    if provider == "mock" and "llm_model" not in overrides:
        overrides["llm_model"] = MOCK_MODEL_NAME
    if overrides:
        from dataclasses import replace

        settings = replace(settings, **overrides)
    return settings
