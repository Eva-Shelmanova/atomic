"""
Базовый класс агента и типы ответов.

Chain of Responsibility: агент либо обрабатывает запрос, либо передаёт дальше.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class AgentType(str, Enum):
    """Типы агентов в цепочке."""
    RETRIEVER = "retriever"
    COMMAND = "command"
    ANALYST = "analyst"


@dataclass
class AgentResponse:
    """Ответ агента."""
    content: str
    agent_type: AgentType
    handled: bool  # True если агент обработал запрос
    metadata: Optional[dict[str, Any]] = None


class BaseAgent(ABC):
    """Базовый агент с паттерном Chain of Responsibility."""

    agent_type: AgentType

    @abstractmethod
    def can_handle(self, query: str, context: dict[str, Any]) -> bool:
        """Проверяет, может ли агент обработать запрос."""
        pass

    @abstractmethod
    def process(self, query: str, context: dict[str, Any]) -> AgentResponse:
        """Обрабатывает запрос или возвращает handled=False для передачи дальше."""
        pass
