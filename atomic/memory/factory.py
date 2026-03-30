"""
Фабрика создания векторной памяти.
"""

from atomic.config import USE_PGVECTOR
from atomic.memory.pgvector_memory import PgVectorMemory
from atomic.memory.vector import VectorMemory


def create_vector_memory():
    """
    Создаёт векторную память в зависимости от конфигурации.
    ATOMIC_USE_PGVECTOR=true → PostgreSQL + pgvector
    иначе → in-memory (для тестов без БД)
    """
    if USE_PGVECTOR:
        return PgVectorMemory()
    return VectorMemory()
