"""
Демо-документы для RAG при старте CLI/API.

Включается только если ATOMIC_SEED_DEMO=true.
Перед вставкой проверяется has_content() — повторные запуски не плодят дубликаты.
"""

from typing import Any, Protocol

from atomic.config import SEED_DEMO


class _VectorMemoryLike(Protocol):
    def has_content(self, content: str) -> bool: ...
    def add(self, content: str, metadata: dict | None = None) -> None: ...


DEMO_DOCUMENTS: list[tuple[str, dict[str, Any]]] = [
    ("atomic — AI-агентная система с роутингом.", {"source": "demo_seed"}),
    ("RetrieverAgent отвечает за поиск в знаниях.", {"source": "demo_seed"}),
    ("CommandAgent выполняет код и действия.", {"source": "demo_seed"}),
]


def seed_demo_vector_memory(vector_memory: _VectorMemoryLike) -> None:
    """Добавляет демо в векторную память, если включён флаг и записи ещё нет."""
    if not SEED_DEMO:
        return
    for content, meta in DEMO_DOCUMENTS:
        if vector_memory.has_content(content):
            continue
        vector_memory.add(content, metadata=meta)
