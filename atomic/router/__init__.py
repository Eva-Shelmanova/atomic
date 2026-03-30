"""
RouterAgent — оркестратор, распределяющий задачи по специализированным агентам.

Функции:
- intent detection
- tool selection
- agent routing
- context preparation
"""

from .agent import RouterAgent

__all__ = ["RouterAgent"]
