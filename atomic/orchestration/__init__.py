"""
Orchestration — workflow engine для координации агентов.

Использует Chain of Responsibility:
Router → Retriever → Command → Analyst
"""

from .workflow import AtomicWorkflow

__all__ = ["AtomicWorkflow"]
