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
from atomic.agents.base import AgentResponse
from atomic.agents.command import CommandAgent
from atomic.agents.retriever import RetrieverAgent
from atomic.contracts.verification import EvidenceKind, PolicyEffect, VerificationBundle
from atomic.memory.chat import ChatMemory
from atomic.memory.factory import create_vector_memory
from atomic.memory.scenario_kb import find_scenario_docs
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

    # --- диагностический контур (этап C) ---
    def _diagnostic_pipeline(self):
        if getattr(self, "_pipeline", None) is None:
            from atomic.diagnostics.pipeline import DiagnosticPipeline

            self._pipeline = DiagnosticPipeline.load()
        return self._pipeline

    def diagnose(
        self,
        source,
        t: float,
        query: str = "Что происходит на станции?",
        use_llm: bool = False,
        session_id: str | None = None,
        doc_min_score: float = 0.6,
    ) -> VerifiedAnswer:
        """Окно телеметрии → гипотезы → проверка → ответ AnalystAgent → PolicyDecision (требования 8–10).

        source — источник телеметрии только для чтения (TelemetryReplay). Диагностический ответ
        всегда проходит верификатор ответа в режиме ENFORCE (D-014), независимо от режима чата.
        Для установленного сценария к ответу добавляется справка из базы знаний (D-040, D-043):
        у pgvector — с косинусной близостью не ниже doc_min_score, у BM25 — с ненулевой оценкой.
        Если пояснение LLM приводит к BLOCK, оно отбрасывается и проверяется детерминированный ответ (D-041).
        """
        from atomic.diagnostics.config import default_config
        from atomic.observability.langfuse import observe_span, trace_ask

        pipeline = self._diagnostic_pipeline()
        cfg = default_config()
        window = source.window_at(t)
        episode = source.episode_for(window, pipeline)
        resolver = EvidenceResolver()
        resolver.register(EvidenceKind.TELEMETRY, source.resolves)
        if pipeline.verifier.analogues is not None:
            resolver.register(EvidenceKind.HISTORICAL_EVENT, pipeline.verifier.analogues.has_event)
        if hasattr(self.vector_memory, "has_id"):
            resolver.register(EvidenceKind.RAG_CHUNK, self.vector_memory.has_id)
        verifier = AnswerVerifier(resolver=resolver)

        self.chat_memory.add("user", query)
        with trace_ask(query, session_id=session_id, name="atomic.diagnose") as span:
            report = pipeline.run(window, episode)
            observe_span("diagnostics.hypotheses", output=report.hypotheses.to_dict(),
                         metadata={"generator": report.versions.get("generator")})
            observe_span("diagnostics.verification", output=[v.to_dict() for v in report.verdicts],
                         metadata={"verifier": report.versions.get("verifier"), "episode_memory": report.episode_memory})
            observe_span("diagnostics.novelty", output=report.novelty.to_dict())
            docs = None
            if report.selected is not None:
                s = report.selected.scenario
                docs = find_scenario_docs(self.vector_memory, s, cfg.scenario_titles.get(s, ""), doc_min_score)
                observe_span("diagnostics.documents", output=[d.get("id") for d in docs])
            t_range = (window.t_start, window.t_end)
            response = self.analyst.explain_diagnosis(report, cfg.scenario_titles, t_range, use_llm=use_llm, docs=docs)
            bundle = verifier.verify(query, response)
            rejected = None
            if use_llm and bundle.decision.effect is PolicyEffect.BLOCK:
                # пояснение LLM не прошло проверку → выпускается только детерминированная часть (D-041)
                rejected = bundle
                stats, narrative = response.metadata.get("narrative_stats"), response.metadata.get("narrative_text")
                response = self.analyst.explain_diagnosis(report, cfg.scenario_titles, t_range, use_llm=False, docs=docs)
                violated = ", ".join(rejected.decision.violated_rule_ids)
                response.content += f"\n\n[Пояснение языковой модели не выпущено: верификатор ответа заблокировал его ({violated}).]"
                response.metadata.update(narrative_stats=stats, narrative_text=narrative, narrative_rejected=rejected.to_dict())
                bundle = verifier.verify(query, response)
            answer = release_text(response.content, bundle)
            if span is not None:
                span.update(output=answer, metadata={
                    "verification_mode": "enforce", "verification": bundle.to_dict(),
                    "diagnosis_outcome": report.outcome.value, "versions": report.versions,
                    "latency_ms": report.latency_ms, "window_id": window.window_id,
                    "narrative_rejected": None if rejected is None else rejected.to_dict(),
                })
        self.chat_memory.add("assistant", answer)
        self.last_diagnosis = report
        return VerifiedAnswer(answer=answer, response=response, bundle=bundle, mode=VerificationMode.ENFORCE)

    def ask(self, query: str, session_id: str | None = None) -> str:
        """
        Основной метод: принять запрос → обработать → сохранить в память → вернуть ответ.
        session_id — для Langfuse Sessions (группировка трасс).
        """
        return self.ask_verified(query, session_id=session_id).answer
