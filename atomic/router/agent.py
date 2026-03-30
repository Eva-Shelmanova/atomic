"""
RouterAgent — оркестратор.

Определяет intent через LLM, выбирает агента и подготавливает контекст.
"""

from enum import Enum
from typing import Any

from atomic.agents.base import AgentType
from atomic.config import MODEL_ROUTER
from atomic.llm.client import LLMClient


class Intent(str, Enum):
    """Типы намерений пользователя."""
    RETRIEVAL = "retrieval"      # поиск в знаниях, RAG
    COMMAND = "command"          # выполнение кода, действий
    ANALYSIS = "analysis"        # рассуждения, планирование
    GENERAL = "general"          # общий вопрос


INTENT_MAP = {
    "retrieval": Intent.RETRIEVAL,
    "command": Intent.COMMAND,
    "analysis": Intent.ANALYSIS,
    "general": Intent.GENERAL,
}


class RouterAgent:
    """
    Роутер: classify_request (LLM) → select_agent → prepare_context.
    """

    ROUTER_PROMPT = """Определи намерение пользователя по запросу. Ответь ОДНИМ словом: retrieval, command, analysis или general.

retrieval — поиск в базе знаний, инструкций, документации. Примеры: "как запустить", "найди информацию", "что известно о", "инструкция по"
command — выполнение КОДА на Python/bash. Только если явно просят выполнить код: "выполни код", "запусти код", "run: print(1+1)"
analysis — анализ, планирование, объяснения: "проанализируй", "объясни", "спланируй"
general — общий вопрос, разговор

Важно: "Как запустить систему?" = retrieval (инструкции). "Выполни код print(1)" = command.

Запрос: {query}

Ответ (одно слово):"""

    def __init__(self, llm_client: LLMClient | None = None):
        self.llm = llm_client or LLMClient(model=MODEL_ROUTER)

    def classify_request(self, query: str) -> Intent:
        """Определяет intent через LLM."""
        prompt = self.ROUTER_PROMPT.format(query=query.strip())
        try:
            response = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=20,
            )
            intent_str = response.lower().strip().split()[0] if response else "general"
            return INTENT_MAP.get(intent_str, Intent.GENERAL)
        except Exception:
            return Intent.GENERAL

    def select_agent(self, intent: Intent) -> AgentType:
        """Выбирает агента по intent."""
        mapping = {
            Intent.RETRIEVAL: AgentType.RETRIEVER,
            Intent.COMMAND: AgentType.COMMAND,
            Intent.ANALYSIS: AgentType.ANALYST,
            Intent.GENERAL: AgentType.ANALYST,
        }
        return mapping[intent]

    def prepare_context(self, query: str, chat_history: list, **kwargs: Any) -> dict[str, Any]:
        """Подготавливает контекст для агентов."""
        return {
            "query": query,
            "chat_history": chat_history or [],
            "intent": self.classify_request(query),
            **kwargs,
        }
