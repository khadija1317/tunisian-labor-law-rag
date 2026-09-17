"""
src/agents/errors.py

Shared exception for LLM call failures (Groq API errors, malformed
JSON that fails pydantic validation after all retries, etc).

Distinguishing this from a legitimate model output matters: without
it, an exhausted-retry fallback and a real "out_of_scope" routing
decision, or a real "ungrounded" verdict, are indistinguishable in
the dispatcher's output -- which would corrupt eval scoring by
silently counting infrastructure failures as correct decisions.
"""


class LLMCallError(Exception):
    """Raised when an LLM call exhausts max_attempts without a valid response."""
    pass