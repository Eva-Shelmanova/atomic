"""
AtomicWorkflow — оркестрация Chain of Responsibility.

Router → Retriever → Command → Analyst → [верификатор ответа] → Memory → Final Answer

Верификатор ответа (этап B3) стоит между _route_and_process() и записью ответа
в память. Режим задаётся ATOMIC_VERIFICATION_MODE (D-014):
    off     — поведение 6-го семестра, проверка не выполняется;
    shadow  — решение считается и пишется в трассу, текст ответа не меняется (по умолчанию);
    enforce — BLOCK заменяет ответ безопасной заглушкой, REQUIRE_HUMAN добавляет пометку.
"""

from dataclasses import dataclass

from atomic.agents.analyst import AnalystAgent
from atomic.agents.base import AgentResponse, AgentType
from atomic.agents.command import CommandAgent
from atomic.agents.retriever import RetrieverAgent
from atomic.contracts.verification import EvidenceKind, VerificationBundle
from atomic.memory.chat import ChatMemory
from atomic.memory.factory import create_vector_memory
from atomic.router.agent import RouterAgent
from atomic.tools.code_executor import CodeExecutor
from atomic.verification import AnswerVerifier, EvidenceResolver, VerificationMode, release_text


@dataclass
class VerifiedAnswer:
    """Результат ask_verified(): выпущенный текст, сырой ответ агента и пакет проверки."""

    answer: str
    response: AgentResponse
    bundle: VerificationBundle | None
    mode: VerificationMode


class AtomicWorkflow:
    """
    Workflow engine: роутинг + цепочка агентов + верификатор ответа + память.
    """

    def __init__(
        self,
        chat_memory: ChatMemory | None = None,
        vector_memory=None,
        code_executor: CodeExecutor | None = None,
        verifier: AnswerVerifier | None = None,
        verification_mode: VerificationMode | str | None = None,
    ):
        self.router = RouterAgent()
        self.chat_memory = chat_memory or ChatMemory()
        self.vector_memory = vector_memory or create_vector_memory()
        self.code_executor = code_executor or CodeExecutor()

        # Chain of Responsibility: порядок имеет значение
        self.retriever = RetrieverAgent(vector_memory=self.vector_memory)
        self.command = CommandAgent(code_executor=self.code_executor)
        self.analyst = AnalystAgent()

        self.verification_mode = (
            VerificationMode(verification_mode) if verification_mode else VerificationMode.from_env()
        )
        self.verifier = verifier or AnswerVerifier(resolver=self._default_resolver())

    def _default_resolver(self) -> EvidenceResolver:
        """Ссылки разрешаются по артефактам этого workflow: база знаний и журнал запусков кода."""
        resolver = EvidenceResolver()
        if hasattr(self.vector_memory, "has_id"):
            resolver.register(EvidenceKind.RAG_CHUNK, self.vector_memory.has_id)
        resolver.register(EvidenceKind.TOOL_OUTPUT, self.code_executor.has_run)
        return resolver

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

    def _release(self, query: str, response: AgentResponse) -> tuple[str, VerificationBundle | None]:
        if self.verification_mode is VerificationMode.OFF:
            return response.content, None
        bundle = self.verifier.verify(query, response)
        if self.verification_mode is VerificationMode.ENFORCE:
            return release_text(response.content, bundle), bundle
        return response.content, bundle

    def ask_verified(self, query: str, session_id: str | None = None) -> VerifiedAnswer:
        """Запрос → агент → верификатор ответа → память. Возвращает и текст, и пакет проверки."""
        from atomic.observability.langfuse import trace_ask

        self.chat_memory.add("user", query)
        with trace_ask(query, session_id=session_id) as span:
            response = self._route_and_process(query)
            answer, bundle = self._release(query, response)
            if span is not None:
                metadata = {"verification_mode": self.verification_mode.value}
                if bundle is not None:
                    metadata["verification"] = bundle.to_dict()
                span.update(output=answer, metadata=metadata)
        # В память попадает только выпущенный текст (заглушка вместо заблокированного ответа).
        self.chat_memory.add("assistant", answer)
        return VerifiedAnswer(answer=answer, response=response, bundle=bundle, mode=self.verification_mode)

    def ask(self, query: str, session_id: str | None = None) -> str:
        """
        Основной метод: принять запрос → обработать → сохранить в память → вернуть ответ.
        session_id — для Langfuse Sessions (группировка трасс).
        """
        return self.ask_verified(query, session_id=session_id).answer
