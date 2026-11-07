"""Provider-independent LLM interface.

The pipeline only ever talks to ``LLMClient``. Concrete implementations:
  * ``LangChainLLMClient`` – wraps any LangChain chat model (Gemini, OpenAI, ...).
  * ``MockLLMClient``      – deterministic offline fixtures for development/tests.

``task`` and ``payload`` let the *mock* produce task-appropriate fixtures; real
LLM clients ignore them and only see the rendered prompt messages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class LLMResponse:
    text: str
    model_name: str
    provider: str
    latency_s: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None
    is_mock: bool = False
    raw: dict[str, Any] = field(default_factory=dict)


class LLMError(RuntimeError):
    """Raised when the provider call itself fails (network, auth, quota...)."""


class LLMUnavailableError(LLMError):
    """Raised when a live LLM is requested but not configured (e.g. no API key)."""


@runtime_checkable
class LLMClient(Protocol):
    provider: str
    model_name: str
    is_mock: bool

    def invoke(self, messages: list[ChatMessage], *, task: str, payload: dict[str, Any] | None = None) -> LLMResponse:
        ...
