"""
Встраивание верификатора ответа в AtomicWorkflow без LLM (этап B3).

LLM роутера и агентов подменяется детерминированной заглушкой, поэтому тест
проверяет именно логику шлюза: режимы off / shadow / enforce, запись в память.
"""


from atomic.agents.base import AgentResponse, AgentType
from atomic.contracts.verification import PolicyEffect
from atomic.memory.vector import VectorMemory
from atomic.orchestration.workflow import AtomicWorkflow
from atomic.tools.code_executor import CodeExecutor


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply

    def chat(self, messages, temperature=0.3, max_tokens=2048):
        return self.reply(messages[-1]["content"]) if callable(self.reply) else self.reply


def route(prompt):
    q = prompt.split("Запрос:")[-1].lower()
    if "выполни код" in q:
        return "command"
    if "найди" in q:
        return "retrieval"
    return "analysis"


def make_workflow(mode):
    vm = VectorMemory()
    vm.add("atomic — AI-агентная система с роутингом.")
    w = AtomicWorkflow(vector_memory=vm, code_executor=CodeExecutor(), verification_mode=mode)
    w.router.llm = FakeLLM(route)
    w.retriever.llm = FakeLLM("atomic — это AI-агентная система.")
    w.analyst.llm = FakeLLM("Архитектура состоит из роутера и агентов.")
    return w


def test_enforce_command_allowed_and_result_kept():
    w = make_workflow("enforce")
    r = w.ask_verified("выполни код print(1+1)")
    assert r.bundle.decision.effect is PolicyEffect.ALLOW
    assert "2" in r.answer
    assert r.bundle.claims[0].evidence[0].ref_id == w.code_executor.last_run_id


def test_enforce_retrieval_with_docs_allowed():
    w = make_workflow("enforce")
    r = w.ask_verified("найди информацию об atomic")
    assert r.bundle.decision.effect is PolicyEffect.ALLOW
    assert r.bundle.claims[0].evidence[0].ref_id == "mem:1"
    assert r.answer == "atomic — это AI-агентная система."


def test_enforce_retrieval_without_docs_requires_human():
    w = make_workflow("enforce")
    w.retriever.vector_memory = VectorMemory()  # пустая база знаний
    r = w.ask_verified("найди информацию об atomic")
    assert r.bundle.decision.effect is PolicyEffect.REQUIRE_HUMAN


def test_enforce_analyst_requires_human_with_marker():
    w = make_workflow("enforce")
    r = w.ask_verified("проанализируй архитектуру")
    assert r.bundle.decision.effect is PolicyEffect.REQUIRE_HUMAN
    assert r.answer.startswith("[Требуется оценка оператора")
    assert "Архитектура состоит" in r.answer


def test_enforce_blocks_unverifiable_answer_and_memory_gets_stub(monkeypatch):
    w = make_workflow("enforce")
    monkeypatch.setattr(
        w.analyst, "process",
        lambda q, c: AgentResponse(content="СЕКРЕТНЫЙ сырой ответ", agent_type=AgentType.ANALYST, handled=True),
    )
    r = w.ask_verified("проанализируй архитектуру")
    assert r.bundle.decision.effect is PolicyEffect.BLOCK
    assert "СЕКРЕТНЫЙ" not in r.answer
    last = w.chat_memory.get_history(limit=1)[-1]
    assert last["role"] == "assistant" and "СЕКРЕТНЫЙ" not in last["content"]


def test_shadow_keeps_text_but_computes_decision():
    w = make_workflow("shadow")
    r = w.ask_verified("проанализируй архитектуру")
    assert r.answer == "Архитектура состоит из роутера и агентов."
    assert r.bundle.decision.effect is PolicyEffect.REQUIRE_HUMAN


def test_off_skips_verification():
    w = make_workflow("off")
    r = w.ask_verified("проанализируй архитектуру")
    assert r.bundle is None and r.answer == "Архитектура состоит из роутера и агентов."


def test_ask_still_returns_str():
    assert isinstance(make_workflow("shadow").ask("выполни код print(1+1)"), str)


def test_default_mode_is_shadow(monkeypatch):
    monkeypatch.delenv("ATOMIC_VERIFICATION_MODE", raising=False)
    assert AtomicWorkflow(vector_memory=VectorMemory()).verification_mode.value == "shadow"
