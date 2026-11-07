"""LangChain-backed client: one wrapper for every supported provider."""

from __future__ import annotations

import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.rate_limiters import InMemoryRateLimiter

from config import Settings
from llm.base import ChatMessage, LLMError, LLMResponse, LLMUnavailableError


def build_chat_model(provider: str, model: str, settings: Settings):
    """Instantiate the LangChain chat model for ``provider``.

    Provider packages are imported lazily so only the one you use must be installed.
    """
    rate_limiter = None
    if settings.llm_requests_per_minute > 0:
        rate_limiter = InMemoryRateLimiter(
            requests_per_second=settings.llm_requests_per_minute / 60.0,
            check_every_n_seconds=0.1,
            max_bucket_size=1,
        )
    common: dict[str, Any] = {"temperature": settings.llm_temperature, "rate_limiter": rate_limiter}
    try:
        if provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                model=model,
                max_output_tokens=settings.llm_max_output_tokens,
                timeout=settings.llm_timeout_seconds,
                max_retries=settings.llm_max_retries,
                **common,
            )
        if provider == "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(model=model, max_tokens=settings.llm_max_output_tokens, timeout=settings.llm_timeout_seconds, max_retries=settings.llm_max_retries, **common)
        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(model=model, max_tokens=settings.llm_max_output_tokens, timeout=settings.llm_timeout_seconds, max_retries=settings.llm_max_retries, **common)
        if provider == "groq":
            from langchain_groq import ChatGroq

            return ChatGroq(model=model, max_tokens=settings.llm_max_output_tokens, timeout=settings.llm_timeout_seconds, max_retries=settings.llm_max_retries, **common)
        if provider == "ollama":
            import os

            from langchain_ollama import ChatOllama

            return ChatOllama(model=model, base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"), num_predict=settings.llm_max_output_tokens, **common)
    except ImportError as exc:
        pkg = {"gemini": "langchain-google-genai", "openai": "langchain-openai", "anthropic": "langchain-anthropic", "groq": "langchain-groq", "ollama": "langchain-ollama"}[provider]
        raise LLMUnavailableError(f"Provider '{provider}' needs the package '{pkg}'. Install it with: pip install {pkg}") from exc
    raise LLMUnavailableError(f"Unsupported provider '{provider}'")


def _to_lc(messages: list[ChatMessage]):
    mapping = {"system": SystemMessage, "user": HumanMessage, "assistant": AIMessage}
    return [mapping[m.role](content=m.content) for m in messages]


def _content_to_text(content: Any) -> str:
    """Chat models may return a string or a list of content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)


class LangChainLLMClient:
    is_mock = False

    def __init__(self, provider: str, model_name: str, settings: Settings):
        self.provider = provider
        self.model_name = model_name
        self._model = build_chat_model(provider, model_name, settings)

    def invoke(self, messages: list[ChatMessage], *, task: str, payload: dict[str, Any] | None = None) -> LLMResponse:
        start = time.perf_counter()
        try:
            result = self._model.invoke(_to_lc(messages))
        except Exception as exc:  # provider SDKs raise many different exception types
            raise LLMError(f"{self.provider} call failed: {type(exc).__name__}: {exc}") from exc
        latency = time.perf_counter() - start
        usage = getattr(result, "usage_metadata", None) or {}
        return LLMResponse(
            text=_content_to_text(result.content),
            model_name=self.model_name,
            provider=self.provider,
            latency_s=round(latency, 3),
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            is_mock=False,
        )
