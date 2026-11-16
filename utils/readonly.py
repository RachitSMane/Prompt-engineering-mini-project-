"""Opt-in read-only demo mode (``REVIEWIQ_DEMO_READONLY=true``).

When enabled, ReviewIQ only *displays* saved results:

* every persistence helper that writes experiment results, summaries, judgements, annotations,
  exports or the activity log refuses with :class:`ReadOnlyModeError` (see ``utils.storage`` and
  the other writers that call :func:`ensure_writable`);
* no LLM client can be created or invoked (``llm.factory``, ``llm.langchain_client``, ``llm.mock``),
  so no model is loaded and no inference runs.

The UI additionally disables the corresponding buttons, but the guard lives in the backend so that
no code path can bypass it. The flag is read from the environment on every call (``.env`` is loaded
by ``config`` on import), so tests and the app see changes immediately. Unset or false = normal mode.

Diagnostic logging (``logs/pipeline.log``) is not a result file and is not blocked.
"""

from __future__ import annotations

import os

ENV_VAR = "REVIEWIQ_DEMO_READONLY"
TRUE_VALUES = {"1", "true", "yes", "on"}

DEMO_BANNER = "DEMO MODE — READ ONLY. Saved experiment results are protected."
DISABLED_REASON = (
    "Disabled in read-only demo mode: this action would call a language model or write to the saved "
    f"experiment results. Restart the app without {ENV_VAR}=true to use it."
)


class ReadOnlyModeError(PermissionError):
    """A write or an LLM call was attempted while read-only demo mode is enabled."""


def is_readonly() -> bool:
    return os.getenv(ENV_VAR, "").strip().lower() in TRUE_VALUES


def ensure_writable(target: object = "") -> None:
    """Raise ReadOnlyModeError if demo mode is on. Call before any write of a result/annotation file."""
    if is_readonly():
        where = f" ({target})" if target else ""
        raise ReadOnlyModeError(f"Read-only demo mode: refusing to write{where}. {DISABLED_REASON}")


def ensure_inference_allowed(what: str = "LLM call") -> None:
    """Raise ReadOnlyModeError if demo mode is on. Call before creating or invoking any model client."""
    if is_readonly():
        raise ReadOnlyModeError(f"Read-only demo mode: refusing {what}. {DISABLED_REASON}")
