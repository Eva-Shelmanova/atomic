"""
Vector Memory — in-memory хранилище для RAG с поиском BM25 (запасной путь без PostgreSQL).
"""

from typing import Any


class VectorMemory:
    """
    Векторное хранилище для RAG.
    Production: FAISS, Qdrant, Weaviate.
    Демо и тесты: in-memory список, поиск BM25 (до этапа B — поиск по подстроке).
    """

    def __init__(self):
        self._docs: list[dict[str, Any]] = []
        self._counter = 0

    def has_id(self, doc_id: str) -> bool:
        """Есть ли документ с таким id (для разрешения ссылок верификатором ответа)."""
        return any(d.get("id") == doc_id for d in self._docs)

    def has_content(self, content: str) -> bool:
        """Проверка точного совпадения текста (для идемпотентного сида)."""
        return any(d.get("content") == content for d in self._docs)

    def add(self, content: str, metadata: dict | None = None) -> None:
        """Добавляет документ."""
        self._counter += 1
        self._docs.append({
            "id": f"mem:{self._counter}",
            "content": content,
            "metadata": metadata or {},
        })

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Поиск по запросу: BM25 по словам и биграммам (atomic/memory/bm25.py).
        Это запасной путь, когда PostgreSQL + pgvector недоступен;
        production-поиск — эмбеддинги и косинусная близость (PgVectorMemory).
        """
        from atomic.memory.bm25 import BM25

        if not self._docs:
            return []
        index = BM25([d.get("content", "") for d in self._docs])
        return [self._docs[i] for i, _ in index.top(query, k=top_k)]
