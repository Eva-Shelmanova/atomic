"""Контракты данных: инварианты и сериализация (этап A)."""

import numpy as np
import pytest

from atomic.contracts import (
    ChannelEvidence,
    Claim,
    EvidenceDirection,
    EvidenceKind,
    EvidenceRef,
    HypothesisSet,
    ObservationWindow,
    PolicyDecision,
    PolicyEffect,
    Rule,
    RulePack,
    ScenarioHypothesis,
    VerificationBundle,
)


def ref(rid="doc-1", kind=EvidenceKind.RAG_CHUNK, trust=0.8):
    return EvidenceRef(ref_id=rid, kind=kind, trust_base=trust)


def test_evidence_ref_rejects_bad_trust_and_empty_id():
    with pytest.raises(ValueError):
        EvidenceRef(ref_id="x", kind=EvidenceKind.RAG_CHUNK, trust_base=1.5)
    with pytest.raises(ValueError):
        EvidenceRef(ref_id="", kind=EvidenceKind.RAG_CHUNK, trust_base=0.5)


def test_claim_strength_is_min_trust_and_zero_without_evidence():
    assert Claim("c1", "t", (ref(trust=0.9), ref("d2", trust=0.4))).strength() == pytest.approx(0.4)
    assert Claim("c2", "t").strength() == 0.0


def test_rule_severity_validated():
    with pytest.raises(ValueError):
        Rule("X", "fatal")
    pack = RulePack("p-1.0.0", (Rule("A", "error"),))
    assert pack.rule("A").severity == "error"
    with pytest.raises(KeyError):
        pack.rule("B")


def test_bundle_roundtrip():
    b = VerificationBundle(
        query="q",
        claims=(Claim("c1", "текст", (ref(),)),),
        rule_results={"EVIDENCE_REQUIRED": True},
        decision=PolicyDecision(PolicyEffect.ALLOW, "aes-core-1.0.0", (), 0.8, ("ok",)),
        agent="retriever",
        rule_details={"EVIDENCE_REQUIRED": []},
    )
    again = VerificationBundle.from_dict(b.to_dict())
    assert again.to_dict() == b.to_dict()
    assert again.decision.effect is PolicyEffect.ALLOW


def test_hypothesis_set_invariants():
    h1 = ScenarioHypothesis("LOCA", 0.6, 1)
    h2 = ScenarioHypothesis("LOCAC", 0.3, 2)
    hs = HypothesisSet("w1", (h1, h2), "gen-1")
    assert hs.top.scenario == "LOCA" and hs.scenarios() == ["LOCA", "LOCAC"]
    with pytest.raises(ValueError):  # не по убыванию
        HypothesisSet("w1", (h2, ScenarioHypothesis("LOCA", 0.6, 2)), "gen-1")
    with pytest.raises(ValueError):  # сумма > 1
        HypothesisSet("w1", (ScenarioHypothesis("A", 0.7, 1), ScenarioHypothesis("B", 0.6, 2)), "g")
    with pytest.raises(ValueError):  # уверенность вне [0, 1]
        ScenarioHypothesis("A", 1.2, 1)


def test_hypothesis_text_has_no_action_recommendation():
    text = ScenarioHypothesis("LOCA", 0.62, 1).describe({"LOCA": "течь"})
    assert text.startswith("Вероятна ситуация LOCA")
    for imperative in ("выполните", "откройте", "закройте", "необходимо", "рекоменд"):
        assert imperative not in text.lower()


def test_channel_evidence_to_ref():
    ev = ChannelEvidence("P", "w1", EvidenceDirection.SUPPORTS, -2.0, -1.5, 0.8)
    r = ev.to_evidence_ref(0.9)
    assert r.ref_id == "w1#P" and r.kind is EvidenceKind.TELEMETRY and r.locator == "P"


def test_observation_window_shape_checked():
    with pytest.raises(ValueError):
        ObservationWindow("w", "r", 0, 10, ("P",), np.zeros((3, 2)), np.zeros((3, 2)))
