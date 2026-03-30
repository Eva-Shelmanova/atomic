"""
Langfuse — трейсинг LLM вызовов.

Включение: задать LANGFUSE_SECRET_KEY и LANGFUSE_PUBLIC_KEY.
Session: передать session_id в trace_ask() для группировки трасс в Langfuse.
"""

from contextlib import contextmanager
from typing import Any, Generator, Optional

from atomic.config import LANGFUSE_BASE_URL, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY


def _is_enabled() -> bool:
    return bool(LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY)


def _get_client():
    if not _is_enabled():
        return None
    try:
        from langfuse import Langfuse

        return Langfuse(
            secret_key=LANGFUSE_SECRET_KEY,
            public_key=LANGFUSE_PUBLIC_KEY,
            base_url=LANGFUSE_BASE_URL,
        )
    except Exception:
        return None


@contextmanager
def trace_ask(query: str, session_id: Optional[str] = None) -> Generator[Any, None, None]:
    """Контекст-менеджер для трейса полного запроса ask().
    session_id — для группировки трасс в Langfuse Sessions (до 200 символов).
    """
    client = _get_client()
    if not client:
        yield None
        return

    from langfuse import propagate_attributes

    # propagate_attributes должен оборачивать весь блок, чтобы session_id попал во все дочерние spans
    with client.start_as_current_observation(
        name="atomic.ask",
        as_type="span",
        input={"query": query},
    ) as span:
        try:
            if session_id:
                with propagate_attributes(session_id=session_id[:200]):
                    yield span
            else:
                yield span
        finally:
            try:
                client.flush()
            except Exception:
                pass


def observe_llm(
    model: str,
    messages: list[dict],
    output: str,
    metadata: dict | None = None,
) -> None:
    """Записывает LLM generation в текущий trace."""
    client = _get_client()
    if not client:
        return

    try:
        with client.start_as_current_observation(
            name="llm.chat",
            as_type="generation",
            model=model,
            input=messages,
            output=output,
            metadata=metadata or {},
        ):
            pass
    except Exception:
        pass


def observe_embedding(model: str, input_texts: list[str], metadata: dict | None = None) -> None:
    """Записывает embedding call в текущий trace."""
    client = _get_client()
    if not client:
        return

    try:
        with client.start_as_current_observation(
            name="embedding",
            as_type="embedding",
            input={"texts": input_texts[:3], "count": len(input_texts)},
            metadata={"model": model, **(metadata or {})},
        ):
            pass
    except Exception:
        pass
