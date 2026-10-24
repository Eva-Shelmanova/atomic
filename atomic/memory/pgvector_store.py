"""
PostgreSQL + pgvector — векторное хранилище для RAG.
"""

import json
from typing import Any, Optional

from atomic.config import EMBEDDING_DIM, PG_VECTOR_DSN


class PgVectorStore:
    """
    Векторное хранилище на PostgreSQL с расширением pgvector.
    """

    def __init__(
        self,
        dsn: Optional[str] = None,
        table_name: str = "atomic_documents",
        embedding_dim: int = EMBEDDING_DIM,
    ):
        self.dsn = dsn or PG_VECTOR_DSN
        self.table_name = table_name
        self.embedding_dim = embedding_dim
        self._conn = None

    def _get_conn(self):
        import psycopg2
        from pgvector.psycopg2 import register_vector

        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self.dsn)
            self._conn.autocommit = True
            with self._conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            register_vector(self._conn)
        return self._conn

    def init_schema(self) -> None:
        """Создаёт таблицу с векторной колонкой."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id SERIAL PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata JSONB DEFAULT '{{}}',
                    embedding vector({self.embedding_dim})
                );
            """)
            # HNSW индекс (pgvector 0.5+), работает с пустой таблицей
            try:
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.table_name}_embedding
                    ON {self.table_name} USING hnsw (embedding vector_cosine_ops);
                """)
            except Exception:
                pass  # индекс опционален, поиск будет медленнее

    def content_exists(self, content: str) -> bool:
        """True, если уже есть строка с точно таким content (без дубликатов при сиде)."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT 1 FROM {self.table_name} WHERE content = %s LIMIT 1",
                (content,),
            )
            return cur.fetchone() is not None

    def add(self, content: str, embedding: list[float], metadata: Optional[dict] = None) -> None:
        """Добавляет документ с эмбеддингом."""
        import numpy as np

        conn = self._get_conn()
        meta_json = json.dumps(metadata or {})
        vec = np.array(embedding, dtype=np.float32)
        with conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO {self.table_name} (content, metadata, embedding) VALUES (%s, %s, %s)",
                (content, meta_json, vec),
            )

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict[str, Any]]:
        """
        Поиск по косинусному сходству.
        Возвращает список [{"content": ..., "metadata": ..., "score": ...}]
        """
        import numpy as np

        conn = self._get_conn()
        vec = np.array(query_embedding, dtype=np.float32)
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, content, metadata, 1 - (embedding <=> %s) as score
                FROM {self.table_name}
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (vec, vec, top_k),
            )
            rows = cur.fetchall()
        return [
            {"id": f"{self.table_name}:{r[0]}", "content": r[1], "metadata": r[2] or {}, "score": float(r[3])}
            for r in rows
        ]

    def id_exists(self, doc_id: str) -> bool:
        """doc_id вида '<таблица>:<id>' — для разрешения ссылок верификатором ответа."""
        prefix = f"{self.table_name}:"
        if not doc_id.startswith(prefix) or not doc_id[len(prefix):].isdigit():
            return False
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(f"SELECT 1 FROM {self.table_name} WHERE id = %s LIMIT 1", (int(doc_id[len(prefix):]),))
            return cur.fetchone() is not None

    def close(self) -> None:
        """Закрывает соединение."""
        if self._conn and not self._conn.closed:
            self._conn.close()
            self._conn = None
