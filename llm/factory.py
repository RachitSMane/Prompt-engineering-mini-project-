"""Factory that returns the right ``LLMClient`` for the current configuration."""

from __future__ import annotations

from config import Settings
from llm.base import LLMClient, LLMUnavailableError


def get_llm_client(settings: Settings, *, use_mock: bool | None = None, role: str = "extraction") -> LLMClient:
    """Return a live client, or the mock client when explicitly requested.

    Live use never silently falls back to mock: if no API key is configured an
    ``LLMUnavailableError`` is raised with a helpful message.

    role: "extraction" (also used for summaries) or "judge".
    """
    use_mock = settings.is_mock if use_mock is None else use_mock
    if use_mock:
        from llm.mock import MockLLMClient

        return MockLLMClient(malformed_rate=settings.mock_malformed_rate)

    provider = settings.effective_judge_provider if role == "judge" else settings.llm_provider
    model = settings.effective_judge_model if role == "judge" else settings.llm_model
    if not settings.live_llm_available(provider):
        raise LLMUnavailableError(settings.live_unavailable_reason(provider))

    from llm.langchain_client import LangChainLLMClient

    return LangChainLLMClient(provider, model, settings)
