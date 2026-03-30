"""
AnalystAgent — агент рассуждений.

Назначение: analysis, planning, explanation, decision support.
"""

from atomic.agents.base import AgentResponse, AgentType, BaseAgent
from atomic.config import MODEL_ANALYST
from atomic.llm.client import LLMClient


class AnalystAgent(BaseAgent):
    """
    Агент анализа и объяснений через LLM.
    Обрабатывает: architecture design, data analysis, strategy, общие вопросы.
    """

    agent_type = AgentType.ANALYST

    ANALYST_PROMPT = """Ты — аналитик и эксперт. Ответь на вопрос пользователя: рассуждай, анализируй, давай рекомендации.
Будь конкретным и полезным. Отвечай на том же языке, что и вопрос.

Вопрос: {query}

Ответ:"""

    def __init__(self, llm_client: LLMClient | None = None):
        self.llm = llm_client or LLMClient(model=MODEL_ANALYST)

    def can_handle(self, query: str, context: dict) -> bool:
        return True

    def process(self, query: str, context: dict) -> AgentResponse:
        chat_history = context.get("chat_history", [])
        # Формируем контекст из истории (последние 4 сообщения)
        history_text = ""
        for msg in chat_history[-4:]:
            role = "Пользователь" if msg.get("role") == "user" else "Ассистент"
            history_text += f"{role}: {msg.get('content', '')}\n"

        prompt = self.ANALYST_PROMPT.format(query=query.strip())
        if history_text:
            prompt = f"Предыдущий контекст:\n{history_text}\n\n{prompt}"

        content = self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=2048,
        )

        return AgentResponse(
            content=content or "Не удалось сформировать ответ.",
            agent_type=self.agent_type,
            handled=True,
            metadata={},
        )
