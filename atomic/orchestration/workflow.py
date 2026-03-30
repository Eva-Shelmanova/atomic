"""
AtomicWorkflow — оркестрация Chain of Responsibility.

Router → Retriever → Command → Analyst → Memory → Final Answer
"""

from atomic.agents.analyst import AnalystAgent
from atomic.agents.base import AgentResponse, AgentType
from atomic.agents.command import CommandAgent
from atomic.agents.retriever import RetrieverAgent
from atomic.memory.chat import ChatMemory
from atomic.memory.factory import create_vector_memory
from atomic.router.agent import RouterAgent
from atomic.tools.code_executor import CodeExecutor


class AtomicWorkflow:
    """
    Workflow engine: роутинг + цепочка агентов + память.
    """

    def __init__(
        self,
        chat_memory: ChatMemory | None = None,
        vector_memory=None,
        code_executor: CodeExecutor | None = None,
    ):
        self.router = RouterAgent()
        self.chat_memory = chat_memory or ChatMemory()
        self.vector_memory = vector_memory or create_vector_memory()
        self.code_executor = code_executor or CodeExecutor()

        # Chain of Responsibility: порядок имеет значение
        self.retriever = RetrieverAgent(vector_memory=self.vector_memory)
        self.command = CommandAgent(code_executor=self.code_executor)
        self.analyst = AnalystAgent()

    def _get_chain(self) -> list:
        return [self.retriever, self.command, self.analyst]

    def _route_and_process(self, query: str) -> AgentResponse:
        """Роутинг + проход по цепочке до первого обработавшего."""
        context = self.router.prepare_context(
            query,
            self.chat_memory.get_history(limit=10),
        )

        for agent in self._get_chain():
            if agent.can_handle(query, context):
                response = agent.process(query, context)
                if response.handled:
                    return response

        # Fallback: analyst всегда обрабатывает
        return self.analyst.process(query, context)

    def ask(self, query: str, session_id: str | None = None) -> str:
        """
        Основной метод: принять запрос → обработать → сохранить в память → вернуть ответ.
        session_id — для Langfuse Sessions (группировка трасс).
        """
        from atomic.observability.langfuse import trace_ask

        self.chat_memory.add("user", query)
        with trace_ask(query, session_id=session_id) as span:
            response = self._route_and_process(query)
            answer = response.content
            if span is not None:
                span.update(output=answer)
        self.chat_memory.add("assistant", answer)
        return answer
