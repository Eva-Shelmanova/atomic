"""
AnalystAgent — агент рассуждений.

Назначение: analysis, planning, explanation, decision support.
"""

from atomic.agents.base import AgentResponse, AgentType, BaseAgent
from atomic.agents.diagnosis_text import build_diagnosis_text, narrative_claims
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
            metadata={
                # Рассуждение модели без внешних источников: опора только MODEL_GENERATED.
                "verification": {
                    "claims": [
                        {
                            "claim_id": "analyst-1",
                            "text": (content or "")[:500],
                            "evidence": [{"ref_id": "analyst:reasoning", "kind": "model"}],
                        }
                    ]
                }
            },
        )

    DIAGNOSIS_PROMPT = """Ниже — результат автоматической диагностики состояния АЭС с пронумерованными свидетельствами.
Кратко (2–4 предложения) объясни оператору, что показывает проверка: какие признаки подтверждают
или опровергают гипотезы. Пиши по-русски. Используй только приведённые факты.
В конце КАЖДОГО предложения укажи ссылки на свидетельства в квадратных скобках, например [E1], [A1] или [D1].
Не давай рекомендаций и указаний к действию: решение принимает оператор.

{report}

Пояснение:"""

    def explain_diagnosis(
        self, report, titles: dict[str, str], t_range=None, use_llm: bool = False, docs: list[dict] | None = None,
    ) -> AgentResponse:
        """Ответ по DiagnosticReport (требование 8): детерминированный отчёт + необязательное пояснение LLM.

        Каждое утверждение ссылается на окно/канал, событие ETHZ или фрагмент базы знаний (docs); пояснение LLM
        разбирается по предложениям, предложение без ссылок опирается только на модель.
        """
        dt = build_diagnosis_text(report, titles, t_range, docs=docs)
        claims = list(dt.claims)
        content = dt.text
        stats, narrative = None, None
        if use_llm:
            raw = self.llm.chat(
                messages=[{"role": "user", "content": self.DIAGNOSIS_PROMPT.format(report=dt.text)}],
                temperature=0.2,
                max_tokens=600,
            )
            narrative, n_claims, stats = narrative_claims(raw, dt.citations)
            if narrative:
                content += "\n\nПояснение (сформулировано языковой моделью по свидетельствам выше):\n" + narrative
                claims += n_claims
        return AgentResponse(
            content=content,
            agent_type=self.agent_type,
            handled=True,
            metadata={"verification": {"claims": claims}, "diagnosis": report.to_dict(), "narrative_stats": stats,
                      "narrative_text": narrative},
        )
