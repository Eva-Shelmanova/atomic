"""Память о событии и сигнатуры ранней фазы (этап C, D-030, D-031)."""

import numpy as np
import pytest

from atomic.contracts.hypothesis import EpisodeContext, HypothesisSet, ObservationWindow, ScenarioHypothesis, VerdictStatus
from atomic.diagnostics.hypothesis_verifier import HypothesisVerifier
from atomic.diagnostics.signatures import ONSET_FEATURES, ScenarioSignatures, onset_features, window_matrix_features

CH = ("P", "TAVG")
PARAMS = {"min_weight": 0.3, "top_channels": 4, "onset_top_channels": 6, "confirm_threshold": 0.7,
          "reject_threshold": 0.4, "fallback_min_weight": 0.9, "fallback_confirm_threshold": 0.95}


def test_onset_features_causal_nan():
    base = np.zeros(2)
    early = np.arange(5)[:, None] * np.array([1.0, -1.0])  # только 40 с наблюдений
    f = onset_features(base, early, np.arange(2)).reshape(2, 3)
    assert f[0, 0] == pytest.approx(3.0) and f[1, 0] == pytest.approx(-3.0)  # d30 = точка 3
    assert np.isnan(f[:, 1:]).all()  # d60, d120 ещё не наблюдались


@pytest.fixture(scope="module")
def sigs():
    """Два сценария с одинаковым текущим окном, но разной ранней фазой (как LOCA и LOCAC)."""
    rng = np.random.default_rng(0)
    win = {s: window_matrix_features(rng.normal(0, 0.1, size=(200, 12, 2))) for s in ("HOT", "COLD")}
    onset = {}
    for s, sign in (("HOT", 1.0), ("COLD", -1.0)):
        o = rng.normal(0, 0.1, size=(60, 6))
        o[:, 0::3] += sign  # d30 по обоим каналам
        onset[s] = o
    w_sig = ScenarioSignatures.build(win, CH, np.arange(2))
    o_sig = ScenarioSignatures.build(onset, CH, np.arange(2), feature_labels=ONSET_FEATURES)
    return w_sig, o_sig


def setup(sigs, sign):
    w_sig, o_sig = sigs
    v = HypothesisVerifier(w_sig, PARAMS, onset_signatures=o_sig)
    vals = np.random.default_rng(1).normal(0, 0.1, size=(12, 2))
    w = ObservationWindow("w1", "r", 600, 710, CH, vals, vals)
    early = np.zeros((13, 2))
    early[3] = sign
    ep = EpisodeContext("e", "onset-w", CH, np.zeros(2), early)
    hs = HypothesisSet("w1", (ScenarioHypothesis("COLD", 0.6, 1), ScenarioHypothesis("HOT", 0.4, 2)), "g")
    return v, w, ep, hs


def test_onset_memory_resolves_confusable_pair(sigs):
    v, w, ep, hs = setup(sigs, +1.0)
    verdicts = {x.hypothesis.scenario: x for x in v.verify(w, hs, with_analogues=False, episode=ep)}
    assert verdicts["HOT"].status is VerdictStatus.CONFIRMED
    assert verdicts["COLD"].status is VerdictStatus.REJECTED
    assert any(e.window_id == "onset-w" for e in verdicts["HOT"].evidence_for)
    assert "от начала события" in verdicts["HOT"].evidence_for[0].detail or any(
        "от начала события" in e.detail for e in verdicts["HOT"].evidence_for)


def test_without_memory_fallback_is_strict_and_noted(sigs):
    v, w, _, hs = setup(sigs, +1.0)
    verdicts = v.verify(w, hs, with_analogues=False, episode=None)
    assert all(x.status is not VerdictStatus.CONFIRMED for x in verdicts)
    assert all(any("нет памяти" in n for n in x.notes) for x in verdicts)


def test_episode_from_dataset_is_causal():
    from atomic.diagnostics.config import default_config

    if not (default_config().resolve("processed") / "manifest.json").exists():
        pytest.skip("NPPAD не подготовлен")
    from atomic.diagnostics.nppad import NPPADDataset

    ds = NPPADDataset()
    run = ds.split("val")[0]
    ep = ds.episode(run, upto_t=55.0)
    assert ep.horizon_s <= 55.0 and len(ep.early) == 6  # точки 0…50 с
    full = ds.episode(run, upto_t=5000.0)
    assert full.horizon_s == pytest.approx(120.0)
    assert np.allclose(full.baseline, ds.normalize(ds.run_raw(run)[1])[0])


def test_onset_tracker_latches_until_return_to_baseline():
    from atomic.diagnostics.novelty import AnomalyGate
    from atomic.diagnostics.onset import onset_track, stationary_points

    gate = AnomalyGate(channel_index=np.array([0]), channel_names=("P",))
    # стационар 100 → провал до 90 → новый стационар 90 (авария) → возврат к 100 → снова провал
    x = np.r_[np.full(10, 100.0), np.linspace(100, 90, 5), np.full(40, 90.0), np.linspace(90, 100, 5),
              np.full(40, 100.0), np.linspace(100, 95, 3), np.full(5, 95.0)]
    raw = x[:, None]
    st = stationary_points(raw, gate, lookback=6)
    latched = onset_track(st, 0, 30, raw, gate)
    free = onset_track(st, 0, 0)
    assert latched[9] == 9 and latched[30] == 10 and latched[54] == 10  # новый стационар 90 — событие продолжается
    assert free[54] > 10  # без защёлки событие сбросилось бы на ненормальном стационаре
    assert latched[93] == 10 and latched[94] == 94  # сброс: 30 стационарных точек после возврата к 100
    assert latched[-1] == 100  # новое событие со своим началом
    for i in range(len(x)):  # причинность: начало не позже текущей точки и зависит только от прошлого
        assert latched[i] <= i
        assert onset_track(st[: i + 1], 0, 30, raw[: i + 1], gate)[-1] == latched[i]
