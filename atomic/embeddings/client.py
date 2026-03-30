"""
Embedding клиент — векторизация текста через Ollama.
"""

from typing import Optional

from atomic.config import MODEL_EMBEDDING, OLLAMA_BASE_URL


def _trace_embedding(model: str, texts: list[str]) -> None:
    try:
        from atomic.observability.langfuse import observe_embedding

        observe_embedding(model=model, input_texts=texts)
    except Exception:
        pass


class EmbeddingClient:
    """Клиент для генерации эмбеддингов через Ollama."""

    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None):
        self.model = (model or MODEL_EMBEDDING).replace("ollama/", "")
        self.base_url = base_url or OLLAMA_BASE_URL

    def embed(self, text: str) -> list[float]:
        """Возвращает вектор для одного текста."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Возвращает векторы для списка текстов."""
        if not texts:
            return []
        try:
            from ollama import Client

            client = Client(host=self.base_url)
            response = client.embed(model=self.model, input=texts)
            embeddings = response.get("embeddings", [])
            _trace_embedding(self.model, texts)
            return embeddings
        except Exception as e:
            raise RuntimeError(f"Ошибка embedding: {e}") from e
