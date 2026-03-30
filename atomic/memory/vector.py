"""
Vector Memory — семантический поиск (FAISS / in-memory fallback).
"""

from typing import Any


class VectorMemory:
    """
    Векторное хранилище для RAG.
    Production: FAISS, Qdrant, Weaviate.
    Демо: in-memory список с простым поиском по подстроке.
    """

    def __init__(self):
        self._docs: list[dict[str, Any]] = []

    def has_content(self, content: str) -> bool:
        """Проверка точного совпадения текста (для идемпотентного сида)."""
        return any(d.get("content") == content for d in self._docs)

    def add(self, content: str, metadata: dict | None = None) -> None:
        """Добавляет документ."""
        self._docs.append({
            "content": content,
            "metadata": metadata or {},
        })

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Поиск по запросу.
        Демо: простой поиск по вхождению подстроки.
        Production: embedding + cosine similarity.
        """
        q_lower = query.lower()
        scored = []
        for doc in self._docs:
            content = doc.get("content", "")
            if q_lower in content.lower():
                scored.append((1.0, doc))
            else:
                # Частичное совпадение слов
                words = set(q_lower.split()) & set(content.lower().split())
                if words:
                    scored.append((len(words) / max(len(q_lower.split()), 1), doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:top_k]]
