"""
Конфигурация моделей и подключений.

Модели Ollama (без префикса ollama/ в вызовах API).
"""

import os

# LLM модели для агентов
MODEL_ROUTER = os.getenv("ATOMIC_MODEL_ROUTER", "qcwind/qwen2.5-7B-instruct-Q4_K_M:latest")
MODEL_RETRIEVER = os.getenv("ATOMIC_MODEL_RETRIEVER", "deepseek-r1:latest")
MODEL_COMMAND = os.getenv("ATOMIC_MODEL_COMMAND", "deepseek-r1:latest")
MODEL_ANALYST = os.getenv("ATOMIC_MODEL_ANALYST", "qcwind/qwen2.5-7B-instruct-Q4_K_M:latest")

# Embedding модель (nomic-embed-text: 768 dims)
MODEL_EMBEDDING = os.getenv("ATOMIC_MODEL_EMBEDDING", "nomic-embed-text:latest")
EMBEDDING_DIM = int(os.getenv("ATOMIC_EMBEDDING_DIM", "768"))

# Ollama base URL
OLLAMA_BASE_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Langfuse (опционально: LANGFUSE_SECRET_KEY — для включения трейсинга)
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_BASE_URL = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")

# PostgreSQL для векторной БД (docker compose: atomic:atomic@localhost:5432/atomic)
PG_VECTOR_DSN = os.getenv("ATOMIC_PG_VECTOR_DSN", "postgresql://atomic:atomic@localhost:5432/atomic")
USE_PGVECTOR = (
    os.getenv("ATOMIC_USE_PGVECTOR", "true").lower().strip() in ("true", "1", "yes")
)

# Демо-строки в RAG при старте CLI/API (без дубликатов: вставка только если текста ещё нет в БД)
SEED_DEMO = os.getenv("ATOMIC_SEED_DEMO", "false").lower().strip() in ("true", "1", "yes")
