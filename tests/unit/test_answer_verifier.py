"""Верификатор ответа: правила, policy engine, пакет правил (этап B3, требование 9)."""

import copy

import pytest

from atomic.agents.base import AgentResponse, AgentType
from atomic.contracts.verification import Claim, EvidenceKind, EvidenceRef, PolicyEffect, Rule, RulePack
from atomic.verification import AnswerVerifier, EvidenceResolver, decide, load_rulepack, release_text
from atomic.verification.rules import evidence_required, no_model_only, ref_resolvable


def ev(rid, kind=EvidenceKind.RAG_CHUNK, trust=0.7):
    return EvidenceRef(rid, kind, trust)


def resolver_with(*ids):
    return EvidenceResolver().register_ids(EvidenceKind.RAG_CHUNK, set(ids))


# --- правила: положительный и отрицательный пример каждого ---

def test_evidence_required():
    assert evidence_required((Claim("c", "t", (ev("a"),)),), EvidenceResolver())[0]
    ok, why = evidence_required((Claim("c", "t"),), EvidenceResolver())
    assert not ok and "нет ссылок" in why[0]
    ok, why = evidence_required((), EvidenceResolver())
    assert not ok and "не содержит утверждений" in why[0]


def test_no_model_only():
    mixed = Claim("c", "t", (ev("a"), ev("m", EvidenceKind.MODEL_GENERATED, 0.2)))
    assert no_model_only((mixed,), EvidenceResolver())[0]
    only_model = Claim("c", "t", (ev("m", EvidenceKind.MODEL_GENERATED, 0.2),))
    assert not no_model_only((only_model,), EvidenceResolver())[0]


def test_ref_resolvable():
    claims = (Claim("c", "t", (ev("doc:1"),)),)
    assert ref_resolvable(claims, resolver_with("doc:1"))[0]
    ok, why = ref_resolvable(claims, resolver_with("doc:2"))
    assert not ok and "doc:1" in why[0]
    # вид без зарегистрированного разрешителя — неразрешим
    assert not ref_resolvable((Claim("c", "t", (ev("w#P", EvidenceKind.TELEMETRY, 0.9),)),), resolver_with())[0]
    # MODEL_GENERATED — не ссылка на артефакт, правило его не режет
    assert ref_resolvable((Claim("c", "t", (ev("m", EvidenceKind.MODEL_GENERATED, 0.2),)),), resolver_with())[0]


def test_resolver_swallows_checker_errors():
    r = EvidenceResolver().register(EvidenceKind.RAG_CHUNK, lambda _: 1 / 0)
    assert r.resolves(ev("x")) is False


# --- policy engine: сочетания нарушений ---

PACK = RulePack(
    "test-1.0.0",
    (Rule("E1", "error"), Rule("E2", "error"), Rule("W1", "warning")),
    min_confidence=0.5,
)
STRONG = (Claim("c", "t", (ev("a", trust=0.9),)),)
WEAK = (Claim("c", "t", (ev("a", trust=0.3),)),)


@pytest.mark.parametrize(
    "results,claims,effect",
    [
        ({"E1": True, "E2": True, "W1": True}, STRONG, PolicyEffect.ALLOW),
        ({"E1": False, "E2": True, "W1": True}, STRONG, PolicyEffect.BLOCK),
        ({"E1": False, "E2": False, "W1": False}, STRONG, PolicyEffect.BLOCK),
        ({"E1": True, "E2": True, "W1": False}, STRONG, PolicyEffect.REQUIRE_HUMAN),
        ({"E1": True, "E2": True, "W1": True}, WEAK, PolicyEffect.REQUIRE_HUMAN),
        ({"E1": False, "E2": True, "W1": False}, WEAK, PolicyEffect.BLOCK),
        ({"E1": True, "E2": True}, STRONG, PolicyEffect.REQUIRE_HUMAN),  # нет результата правила = нарушение
    ],
)
def test_policy_combinations(results, claims, effect):
    d = decide(PACK, results, claims)
    assert d.effect is effect
    assert d.applied_pack_id == "test-1.0.0"
    assert set(d.violated_rule_ids) == {r for r in ("E1", "E2", "W1") if not results.get(r, False)}


def test_confidence_is_min_over_chain():
    claims = (Claim("a", "t", (ev("1", trust=0.9), ev("2", trust=0.6))), Claim("b", "t", (ev("3", trust=0.8),)))
    assert decide(PACK, {"E1": True, "E2": True, "W1": True}, claims).propagated_confidence == pytest.approx(0.6)


# --- пакет правил ---

def test_default_rulepack_loads():
    pack = load_rulepack("aes-core-1.0.0")
    assert pack.pack_id == "aes-core-1.0.0"
    assert {r.rule_id for r in pack.rules} == {"EVIDENCE_REQUIRED", "NO_MODEL_ONLY", "REF_RESOLVABLE"}
    assert set(pack.params["trust_base"]) == set(EvidenceKind)


def test_rulepack_requires_trust_for_every_kind(tmp_path):
    src = load_rulepack().params  # noqa: F841 — проверяем валидацию на урезанном файле
    text = "pack_id = 'x-1'\n[[rule]]\nrule_id='EVIDENCE_REQUIRED'\nseverity='error'\n[trust_base]\nmodel = 0.2\n"
    (tmp_path / "x-1.toml").write_text(text)
    with pytest.raises(ValueError, match="trust_base"):
        load_rulepack("x-1", directory=tmp_path)


def test_verifier_rejects_pack_with_unknown_rule():
    pack = copy.copy(load_rulepack())
    bad = RulePack("bad-1", (Rule("NO_SUCH_RULE", "error"),), params=pack.params)
    with pytest.raises(ValueError, match="без реализации"):
        AnswerVerifier(pack=bad)


# --- верификатор целиком ---

def response(meta, agent=AgentType.RETRIEVER):
    return AgentResponse(content="ответ", agent_type=agent, handled=True, metadata=meta)


def test_answer_without_verification_metadata_is_blocked():
    b = AnswerVerifier().verify("q", response({}))
    assert b.decision.effect is PolicyEffect.BLOCK
    assert "EVIDENCE_REQUIRED" in b.decision.violated_rule_ids


def test_answer_without_refs_is_blocked():
    meta = {"verification": {"claims": [{"claim_id": "c1", "text": "t", "evidence": []}]}}
    assert AnswerVerifier().verify("q", response(meta)).decision.effect is PolicyEffect.BLOCK


def test_answer_with_resolvable_refs_is_allowed():
    meta = {"verification": {"claims": [{"text": "t", "evidence": [{"ref_id": "doc:1", "kind": "rag_chunk"}]}]}}
    b = AnswerVerifier(resolver=resolver_with("doc:1")).verify("q", response(meta))
    assert b.decision.effect is PolicyEffect.ALLOW
    assert b.decision.propagated_confidence == pytest.approx(0.7)
    assert b.agent == "retriever"


def test_answer_with_dangling_ref_is_blocked():
    meta = {"verification": {"claims": [{"text": "t", "evidence": [{"ref_id": "doc:404", "kind": "rag_chunk"}]}]}}
    b = AnswerVerifier(resolver=resolver_with("doc:1")).verify("q", response(meta))
    assert b.decision.effect is PolicyEffect.BLOCK
    assert b.decision.violated_rule_ids == ("REF_RESOLVABLE",)


def test_model_only_answer_requires_human():
    meta = {"verification": {"claims": [{"text": "t", "evidence": [{"ref_id": "a", "kind": "model"}]}]}}
    b = AnswerVerifier().verify("q", response(meta, AgentType.ANALYST))
    assert b.decision.effect is PolicyEffect.REQUIRE_HUMAN
    assert b.decision.violated_rule_ids == ("NO_MODEL_ONLY",)


def test_agent_cannot_inflate_trust():
    meta = {"verification": {"claims": [{"text": "t", "evidence": [{"ref_id": "a", "kind": "model", "trust_base": 1.0}]}]}}
    b = AnswerVerifier().verify("q", response(meta))
    assert b.claims[0].evidence[0].trust_base == pytest.approx(0.2)


def test_release_text():
    meta = {"verification": {"claims": [{"text": "t", "evidence": [{"ref_id": "a", "kind": "model"}]}]}}
    human = AnswerVerifier().verify("q", response(meta))
    assert release_text("сырой ответ", human).startswith("[Требуется оценка оператора")
    assert release_text("сырой ответ", human).endswith("сырой ответ")
    blocked = AnswerVerifier().verify("q", response({}))
    text = release_text("сырой ответ", blocked)
    assert "сырой ответ" not in text and blocked.decision.applied_pack_id in text and "EVIDENCE_REQUIRED" in text


def test_bundle_is_json_serializable():
    import json

    meta = {"verification": {"claims": [{"text": "t", "evidence": [{"ref_id": "doc:1", "kind": "rag_chunk"}]}]}}
    b = AnswerVerifier(resolver=resolver_with("doc:1")).verify("q", response(meta))
    assert json.loads(json.dumps(b.to_dict(), ensure_ascii=False))["decision"]["effect"] == "allow"
