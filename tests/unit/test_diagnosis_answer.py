"""Итоговое правило выбора, диагностический ответ, правила aes-core-1.1.0 (этап C)."""

import pytest

from atomic.agents.analyst import AnalystAgent
from atomic.agents.diagnosis_text import build_diagnosis_text, narrative_claims
from atomic.contracts.hypothesis import (
    ChannelEvidence,
    DiagnosisOutcome,
    DiagnosticReport,
    EventAnalogue,
    EvidenceDirection,
    HypothesisSet,
    HypothesisVerdict,
    NoveltyAssessment,
    ScenarioHypothesis,
    VerdictStatus,
)
from atomic.contracts.verification import EvidenceKind, PolicyEffect
from atomic.diagnostics.pipeline import select
from atomic.verification import AnswerVerifier, EvidenceResolver, load_rulepack
from atomic.verification.rules import diagnosis_confirmed, no_directives

TITLES = {"LOCA": "течь, горячая нитка", "LOCAC": "течь, холодная нитка", "LLB": "продувка"}


def ev(ch, d=EvidenceDirection.SUPPORTS, wid="w1"):
    return ChannelEvidence(ch, wid, d, 1.0, 0.9, 0.8, f"{ch}: в диапазоне")


def verdict(s, conf, rank, status, ratio, analogues=()):
    sup = (ev("P"), ev("TAVG")) if status is not VerdictStatus.REJECTED else ()
    con = (ev("RM1", EvidenceDirection.CONTRADICTS),) if status is not VerdictStatus.CONFIRMED else ()
    return HypothesisVerdict(ScenarioHypothesis(s, conf, rank), status, ratio, sup, con, analogues)


def novelty(anomaly=True, novel=False):
    return NoveltyAssessment(anomaly, ("P",) if anomaly else (), novel, 0.5, 1.0, 1.0, 2.0, "LOCA", "episode+window")


def report(verdicts, nov):
    hs = HypothesisSet("w1", tuple(v.hypothesis for v in verdicts), "g")
    out, sel, cand = select(list(verdicts), nov, margin=0.1)
    return DiagnosticReport("w1", hs, tuple(verdicts), nov, out, sel, cand, True)


A = (EventAnalogue("ethz:1", "Site", "leak", 5.0),)
V_ID = [verdict("LOCAC", 0.6, 1, VerdictStatus.CONFIRMED, 0.72), verdict("LOCA", 0.3, 2, VerdictStatus.CONFIRMED, 1.0, A),
        verdict("LLB", 0.05, 3, VerdictStatus.REJECTED, 0.0)]


@pytest.mark.parametrize(
    "verdicts,nov,outcome,selected",
    [
        (V_ID, novelty(), DiagnosisOutcome.IDENTIFIED, "LOCA"),
        ([verdict("LOCAC", 0.6, 1, VerdictStatus.CONFIRMED, 0.95), verdict("LOCA", 0.3, 2, VerdictStatus.CONFIRMED, 0.9)],
         novelty(), DiagnosisOutcome.AMBIGUOUS, None),
        ([verdict("LOCAC", 0.6, 1, VerdictStatus.UNCERTAIN, 0.5)], novelty(), DiagnosisOutcome.UNCONFIRMED, None),
        (V_ID, novelty(novel=True), DiagnosisOutcome.NOVEL, None),
        (V_ID, novelty(anomaly=False), DiagnosisOutcome.NO_ANOMALY, None),
    ],
)
def test_selection_rule(verdicts, nov, outcome, selected):
    rep = report(verdicts, nov)
    assert rep.outcome is outcome
    assert (rep.selected.scenario if rep.selected else None) == selected


def test_every_claim_has_evidence_for_every_outcome():
    for verdicts, nov in [(V_ID, novelty()), (V_ID, novelty(novel=True)), (V_ID, novelty(anomaly=False)),
                          ([verdict("LOCAC", 0.6, 1, VerdictStatus.UNCERTAIN, 0.5)], novelty())]:
        dt = build_diagnosis_text(report(verdicts, nov), TITLES)
        assert dt.claims[0]["attributes"]["type"] == "diagnosis"
        assert all(c["evidence"] for c in dt.claims)
        assert "решение о действиях принимает оператор" in dt.text


def test_identified_text_mentions_selected_and_analogues():
    dt = build_diagnosis_text(report(V_ID, novelty()), TITLES, (480, 590))
    assert "Вероятна ситуация LOCA" in dt.text and "[A1]" in dt.text and "окно 480–590 с" in dt.text
    assert any(c["evidence"][0]["kind"] == "historical_event" for c in dt.claims if c["claim_id"].startswith("analogue"))


def test_narrative_parsing():
    cites = {"E1": {"ref_id": "w1#P", "kind": "telemetry"}, "A1": {"ref_id": "ethz:1", "kind": "historical_event"}}
    text, claims, stats = narrative_claims("<think>черновик</think>Давление падает [E1]. Это похоже на прошлое [A1]. Без ссылки.", cites)
    assert "черновик" not in text
    assert [c["evidence"][0]["kind"] for c in claims] == ["telemetry", "historical_event", "model"]
    assert stats == {"sentences": 3, "cited": 2, "unknown_citations": 0}
    _, claims, stats = narrative_claims("Ссылка в никуда [E9].", cites)
    assert stats["unknown_citations"] == 1 and claims[0]["evidence"][0]["ref_id"] == "citation:E9"


def test_new_rules():
    from atomic.contracts.verification import Claim

    ok = Claim("d", "t", (), {"type": "diagnosis", "outcome": "identified"})
    quiet = Claim("d", "t", (), {"type": "diagnosis", "outcome": "no_anomaly"})
    bad = Claim("d", "t", (), {"type": "diagnosis", "outcome": "novel"})
    r = EvidenceResolver()
    assert diagnosis_confirmed((ok, quiet), r)[0] and not diagnosis_confirmed((bad,), r)[0]
    assert no_directives((Claim("c", "В аналогичных случаях наблюдалось открытие клапана."),), r)[0]
    assert not no_directives((Claim("c", "Откройте клапан подпитки."),), r)[0]
    assert not no_directives((Claim("c", "Необходимо снизить мощность."),), r)[0]


def answer_verifier():
    r = EvidenceResolver().register(EvidenceKind.TELEMETRY, lambda ref: ref.split("#")[0] == "w1")
    r.register(EvidenceKind.HISTORICAL_EVENT, lambda ref: ref == "ethz:1")
    return AnswerVerifier(resolver=r)


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply

    def chat(self, messages, temperature=0.3, max_tokens=2048):
        return self.reply


@pytest.mark.parametrize(
    "nov,reply,effect",
    [
        (novelty(), None, PolicyEffect.ALLOW),
        (novelty(anomaly=False), None, PolicyEffect.ALLOW),
        (novelty(novel=True), None, PolicyEffect.REQUIRE_HUMAN),
        (novelty(), "Давление падает [E1]. Аналог подтверждает [A1].", PolicyEffect.ALLOW),
        (novelty(), "Давление падает [E1]. Вероятно, всё серьёзно.", PolicyEffect.REQUIRE_HUMAN),
        (novelty(), "Давление падает [E1]. Откройте клапан [E1].", PolicyEffect.BLOCK),
        (novelty(), "Давление падает [E99].", PolicyEffect.BLOCK),
    ],
)
def test_answer_decisions(nov, reply, effect):
    agent = AnalystAgent(llm_client=FakeLLM(reply))
    resp = agent.explain_diagnosis(report(V_ID, nov), TITLES, use_llm=reply is not None)
    assert answer_verifier().verify("q", resp).decision.effect is effect


def test_rulepack_versions():
    assert load_rulepack().pack_id == "aes-core-1.1.0"
    assert {r.rule_id for r in load_rulepack().rules} >= {"DIAGNOSIS_CONFIRMED", "NO_DIRECTIVES"}
    assert len(load_rulepack("aes-core-1.0.0").rules) == 3


DOCS = [
    {"id": "mem:3", "content": "Необходимо снизить мощность реактора.", "metadata": {"source": "instr.txt"}},
    {"id": "mem:1", "content": "При течи в горячей нитке давление первого контура снижается.", "metadata": {"source": "kb.txt"}},
]


def test_documents_cited_only_when_identified_and_without_directives():
    dt = build_diagnosis_text(report(V_ID, novelty()), TITLES, docs=DOCS)
    assert "[D1]" in dt.text and "mem:1" in dt.text and "mem:3" not in dt.text  # фрагмент-указание пропущен
    doc_claims = [c for c in dt.claims if c["attributes"]["type"] == "document"]
    assert [c["evidence"][0] for c in doc_claims] == [{"ref_id": "mem:1", "kind": "rag_chunk"}]
    assert "[D1]" not in build_diagnosis_text(report(V_ID, novelty(novel=True)), TITLES, docs=DOCS).text

    r = EvidenceResolver().register(EvidenceKind.TELEMETRY, lambda ref: ref.split("#")[0] == "w1")
    r.register(EvidenceKind.HISTORICAL_EVENT, lambda ref: ref == "ethz:1")
    r.register(EvidenceKind.RAG_CHUNK, lambda ref: ref == "mem:1")
    agent = AnalystAgent(llm_client=FakeLLM("Давление снижается [E1], как описано в базе знаний [D1]."))
    resp = agent.explain_diagnosis(report(V_ID, novelty()), TITLES, use_llm=True, docs=DOCS)
    assert AnswerVerifier(resolver=r).verify("q", resp).decision.effect is PolicyEffect.ALLOW
