"""
Pytest fixtures для atomic.
"""

import os

import pytest

# In-memory память для тестов (без PostgreSQL)
os.environ.setdefault("ATOMIC_USE_PGVECTOR", "false")


@pytest.fixture
def workflow():
    """Workflow с in-memory памятью и демо-документами."""
    from atomic.memory.chat import ChatMemory
    from atomic.memory.factory import create_vector_memory
    from atomic.orchestration.workflow import AtomicWorkflow
    from atomic.tools.code_executor import CodeExecutor

    w = AtomicWorkflow(
        chat_memory=ChatMemory(),
        vector_memory=create_vector_memory(),
        code_executor=CodeExecutor(),
    )
    w.vector_memory.add("atomic — AI-агентная система с роутингом.")
    w.vector_memory.add("RetrieverAgent отвечает за поиск в знаниях.")
    w.vector_memory.add("CommandAgent выполняет код и действия.")
    return w
