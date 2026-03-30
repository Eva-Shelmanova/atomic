"""
Chat Memory — контекст диалога (short-term).
"""

from collections import deque
from typing import Optional


class ChatMemory:
    """
    Хранит историю сообщений пользователя и ассистента.
    Ограничение по количеству сообщений для контроля контекста.
    """

    def __init__(self, max_messages: int = 20):
        self._history: deque[dict] = deque(maxlen=max_messages)

    def add(self, role: str, content: str) -> None:
        """Добавляет сообщение в историю."""
        self._history.append({"role": role, "content": content})

    def get_history(self, limit: Optional[int] = None) -> list[dict]:
        """Возвращает историю. limit — последние N сообщений."""
        items = list(self._history)
        if limit:
            items = items[-limit:]
        return items

    def clear(self) -> None:
        """Очищает историю."""
        self._history.clear()
