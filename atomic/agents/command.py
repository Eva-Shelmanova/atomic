"""
CommandAgent — слой действий.

CodeGen (LLM) → Execute → Debug Loop → Review
"""

import re

from atomic.agents.base import AgentResponse, AgentType, BaseAgent
from atomic.config import MODEL_COMMAND
from atomic.llm.client import LLMClient
from atomic.router.agent import Intent


class CommandAgent(BaseAgent):
    """
    Агент выполнения: генерация кода через LLM, запуск, отладка, проверка.
    """

    agent_type = AgentType.COMMAND

    CODE_GEN_PROMPT = """Пользователь просит выполнить код или действие. Сгенерируй ТОЛЬКО код на Python, без пояснений и markdown.
Код должен быть самодостаточным, с print() для вывода результата.
Если запрос неясен — сгенерируй print("Уточните, какой код выполнить").

Запрос: {query}

Код (только Python):"""

    def __init__(self, code_executor=None, llm_client: LLMClient | None = None):
        self.code_executor = code_executor
        self.llm = llm_client or LLMClient(model=MODEL_COMMAND)

    def can_handle(self, query: str, context: dict) -> bool:
        return context.get("intent") == Intent.COMMAND

    def _code_gen(self, query: str) -> str:
        """Генерация кода через LLM с fallback на извлечение из запроса."""
        # Сначала пробуем извлечь явный код из запроса
        m = re.search(r"```(?:python)?\s*\n([\s\S]+?)```", query)
        if m:
            return m.group(1).strip()
        for sep in ["код", "code", ":"]:
            if sep in query.lower():
                idx = query.lower().rfind(sep) + len(sep)
                rest = query[idx:].strip().lstrip(":")
                if rest and len(rest) < 500:
                    return rest.strip()

        # Используем LLM для генерации
        prompt = self.CODE_GEN_PROMPT.format(query=query.strip())
        code = self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1024,
        )
        # Убираем markdown если есть
        if "```" in code:
            m = re.search(r"```(?:python)?\s*\n([\s\S]+?)```", code)
            if m:
                code = m.group(1).strip()
        return code.strip() or 'print("Не удалось сгенерировать код")'

    def _execute(self, code: str) -> tuple[str, bool]:
        """Выполнение кода."""
        if self.code_executor:
            return self.code_executor.run(code)
        return ("Sandbox не настроен.", False)

    def _review(self, code: str, result: str, success: bool) -> str:
        """Краткая проверка результата."""
        if success:
            return "Выполнение успешно."
        return "Выполнение завершилось с ошибкой. Проверьте код."

    def process(self, query: str, context: dict) -> AgentResponse:
        if not self.can_handle(query, context):
            return AgentResponse(
                content="",
                agent_type=self.agent_type,
                handled=False,
            )

        code = self._code_gen(query)
        output, success = self._execute(code)
        review = self._review(code, output, success)

        content = f"Сгенерированный код:\n```python\n{code}\n```\n\nРезультат: {output}\n\n{review}"
        return AgentResponse(
            content=content,
            agent_type=self.agent_type,
            handled=True,
            metadata={"code": code, "success": success, "output": output},
        )
