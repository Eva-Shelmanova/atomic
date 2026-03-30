"""
PgVectorMemory — векторная память на PostgreSQL с эмбеддингами.
"""

from typing import Any, Optional

from atomic.embeddings.client import EmbeddingClient
from atomic.memory.pgvector_store import PgVectorStore


class PgVectorMemory:
    """
    Векторная память: EmbeddingClient + PgVectorStore.
    Интерфейс совместим с VectorMemory (add, search).
    """

    def __init__(
        self,
        embedding_client: Optional[EmbeddingClient] = None,
        pg_store: Optional[PgVectorStore] = None,
        init_schema: bool = True,
    ):
        self.embedding_client = embedding_client or EmbeddingClient()
        self.pg_store = pg_store or PgVectorStore()
        if init_schema:
            self.pg_store.init_schema()

    def has_content(self, content: str) -> bool:
        """Проверка точного совпадения текста (для идемпотентного сида)."""
        return self.pg_store.content_exists(content)

    def add(self, content: str, metadata: Optional[dict] = None) -> None:
        """Добавляет документ: эмбеддинг → сохранение в PG."""
        embedding = self.embedding_client.embed(content)
        self.pg_store.add(content, embedding, metadata)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Поиск: эмбеддинг запроса → поиск в PG. Возвращает [{"content": ..., "metadata": ..., "score": ...}]."""
        query_embedding = self.embedding_client.embed(query)
        return self.pg_store.search(query_embedding, top_k=top_k)
