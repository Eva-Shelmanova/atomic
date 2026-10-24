"""Верификатор гипотез: свидетельства «за»/«против», статусы, контрастные веса (этап B2)."""

from pathlib import Path

import numpy as np
import pytest

from atomic.contracts.hypothesis import (
    EvidenceDirection,
    HypothesisSet,
    ObservationWindow,
    ScenarioHypothesis,
    VerdictStatus,
)
from atomic.diagnostics.analogues import AnalogueFinder
from atomic.diagnostics.config import default_config
from atomic.diagnostics.hypothesis_verifier import HypothesisVerifier
from atomic.diagnostics.signatures import ScenarioSignatures, run_window_features, window_matrix_features

CH = ("P", "TAVG", "NSGA", "NSGB")
PARAMS = {"min_weight": 0.3, "top_channels": 8, "confirm_threshold": 0.6, "reject_threshold": 0.4, "analogues_k": 2}
T = 12


def make_windows(kind, n, rng):
    """A: растёт P; B: растёт NSGA; C: растёт NSGB. Остальное — шум."""
    out = []
    for _ in range(n):
        w = rng.normal(0, 0.2, size=(T, len(CH)))
        j = {"A": 0, "B": 2, "C": 3}[kind]
        w[:, j] += np.linspace(0, 3, T)
        out.append(w)
    return np.array(out)


@pytest.fixture(scope="module")
def sig():
    rng = np.random.default_rng(0)
    feats = {k: window_matrix_features(make_windows(k, 300, rng)) for k in "ABC"}
    return ScenarioSignatures.build(feats, CH, np.arange(len(CH)), metadata={"fingerprint": "test"})


def window(values, wid="w1"):
    return ObservationWindow(wid, "r", 0.0, 110.0, CH, values, values)


def hset(wid, *names):
    n = len(names)
    return HypothesisSet(wid, tuple(ScenarioHypothesis(s, round(0.9 / n, 3), i + 1) for i, s in enumerate(names)), "gen-test")


def test_window_features_shapes_and_run_windows():
    v = np.zeros((3, T, 4))
    v[:, :, 1] = np.arange(T)
    f = window_matrix_features(v)
    assert f.shape == (3, 8)
    assert f[0, 3] == pytest.approx(1.0)  # TAVG.slope
    assert run_window_features(np.zeros((30, 4)), 12, 6).shape == (4, 8)
    assert run_window_features(np.zeros((5, 4)), 12, 6).shape == (0, 8)


def test_true_hypothesis_confirmed_wrong_rejected(sig):
    rng = np.random.default_rng(1)
    v = HypothesisVerifier(sig, PARAMS)
    w = window(make_windows("B", 1, rng)[0])
    verdicts = {x.hypothesis.scenario: x for x in v.verify(w, hset("w1", "B", "C", "A"))}
    assert verdicts["B"].status is VerdictStatus.CONFIRMED
    assert verdicts["C"].status is VerdictStatus.REJECTED
    assert verdicts["A"].status is VerdictStatus.REJECTED
    assert verdicts["B"].support_ratio > verdicts["C"].support_ratio


def test_evidence_refers_to_real_channels_and_window(sig):
    rng = np.random.default_rng(2)
    v = HypothesisVerifier(sig, PARAMS)
    w = window(make_windows("A", 1, rng)[0], wid="win-42")
    for verdict in v.verify(w, hset("win-42", "A", "B")):
        evs = (*verdict.evidence_for, *verdict.evidence_against)
        assert evs
        for e in evs:
            assert e.window_id == "win-42" and e.channel in CH
            assert e.ref_id == f"win-42#{e.channel}"
            assert 0 < e.weight <= 1
        assert all(e.direction is EvidenceDirection.SUPPORTS for e in verdict.evidence_for)
        assert all(e.direction is EvidenceDirection.CONTRADICTS for e in verdict.evidence_against)


def test_contrastive_weights_focus_on_competitor(sig):
    v = HypothesisVerifier(sig, PARAMS)
    pairs = sig.pairs
    w_bc = v.weights("B", ["B", "C"])
    w_ba = v.weights("B", ["B", "A"])
    # против C важны каналы ПГ (NSGA/NSGB), против A — ещё и давление P
    assert w_bc[pairs.index(("P", "slope"))] < w_ba[pairs.index(("P", "slope"))]


def test_unknown_scenario_is_uncertain(sig):
    v = HypothesisVerifier(sig, PARAMS)
    status, ratio, sup, con, notes = v.verify_one("ZZZ", np.zeros(8), "w", ["ZZZ"])
    assert status is VerdictStatus.UNCERTAIN and notes


def test_mismatched_window_rejected(sig):
    v = HypothesisVerifier(sig, PARAMS)
    with pytest.raises(ValueError):
        v.verify(window(np.zeros((T, 4)), "w1"), hset("other", "A"))


def test_no_action_recommendations_in_evidence(sig):
    rng = np.random.default_rng(3)
    v = HypothesisVerifier(sig, PARAMS)
    for verdict in v.verify(window(make_windows("C", 1, rng)[0]), hset("w1", "C", "B")):
        for e in (*verdict.evidence_for, *verdict.evidence_against):
            assert not any(word in e.detail.lower() for word in ("выполните", "откройте", "закройте", "рекоменд"))


def test_analogues_attached(sig):
    events = [{"event_id": "ethz:7", "site": "S", "country": "C", "date": "2000", "origin": "Steam Generator",
               "description": "steam generator tube rupture", "cause": None}]
    finder = AnalogueFinder(events, {"version": "t", "min_score": 0.1, "scenario": {"B": {"query": "steam generator tube", "origin": []}}})
    v = HypothesisVerifier(sig, PARAMS, analogues=finder)
    rng = np.random.default_rng(4)
    verdicts = v.verify(window(make_windows("B", 1, rng)[0]), hset("w1", "B", "A"))
    assert verdicts[0].analogues[0].event_id == "ethz:7"
    assert verdicts[1].analogues == ()
    assert "analogues-t" in verdicts[0].verifier_version


def test_signatures_save_load(sig, tmp_path):
    sig.save(tmp_path)
    again = ScenarioSignatures.load(tmp_path)
    assert again.scenarios == sig.scenarios and np.allclose(again.specificity, sig.specificity)
    assert again.version == sig.version


MODELS = default_config().resolve("models")


@pytest.mark.skipif(
    not ((MODELS / "generator" / "generator.joblib").exists() and (MODELS / "verifier" / "signatures.npz").exists()),
    reason="модели не обучены (scripts/train_generator.py, scripts/train_verifier.py)",
)
def test_real_pipeline_on_val_window():
    """Окно NPPAD → барьер → генератор → верификатор: верная гипотеза подтверждается (val, не test)."""
    from atomic.diagnostics.barrier import InformationBarrier
    from atomic.diagnostics.generator import HypothesisGenerator
    from atomic.diagnostics.labels import LabelStore
    from atomic.diagnostics.nppad import NPPADDataset

    cfg = default_config()
    ds = NPPADDataset()
    labels = LabelStore.load(ds.directory)
    gen = HypothesisGenerator.load(MODELS / "generator")
    ver = HypothesisVerifier(ScenarioSignatures.load(MODELS / "verifier"), cfg.verifier)
    run = next(r for r in ds.split("val") if labels.scenario(r) == "SLBIC")
    w = list(ds.windows(run))[3]
    hs = gen.generate(InformationBarrier().restrict(w))
    verdicts = {v.hypothesis.scenario: v for v in ver.verify(w, hs, with_analogues=False)}
    assert "SLBIC" in verdicts
    assert verdicts["SLBIC"].status is VerdictStatus.CONFIRMED
