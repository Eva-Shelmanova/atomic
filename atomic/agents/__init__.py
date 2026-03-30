"""
Chain of Responsibility — специализированные агенты.

Каждый агент может обработать запрос или передать дальше:
- RetrieverAgent (RAG)
- CommandAgent (действия: CodeGen, Execute, Debug, Review)
- AnalystAgent (рассуждения, анализ)
"""

from .base import BaseAgent, AgentResponse
from .retriever import RetrieverAgent
from .command import CommandAgent
from .analyst import AnalystAgent

__all__ = [
    "BaseAgent",
    "AgentResponse",
    "RetrieverAgent",
    "CommandAgent",
    "AnalystAgent",
]
