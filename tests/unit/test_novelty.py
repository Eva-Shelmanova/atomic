"""Проверка отклонения и новизны (этап C, D-032)."""

import numpy as np
import pytest

from atomic.contracts.hypothesis import EpisodeContext, ObservationWindow
from atomic.diagnostics.novelty import AnomalyGate, NoveltyDetector, fit_novelty

NAMES = ("P", "TAVG", "HTR", "RM2")


def gate(**kw):
    return AnomalyGate(channel_index=np.arange(4), channel_names=NAMES, drift_channels=("HTR",), **kw)


def test_gate_counts_relative_deviation_and_ignores_drift():
    g = gate()
    base = np.array([155.0, 310.0, 100.0, 0.0])
    quiet = np.array([155.5, 310.5, 900.0, 0.0])  # HTR дрейфует, остальное < 1 %
    assert g.check(quiet, base) == (False, [])
    leak = np.array([140.0, 310.0, 100.0, 0.0])  # P упало на ~10 %
    anomaly, dev = g.check(leak, base)
    assert anomaly and dev == ["P"]
    # канал с нулевым исходным значением: отклонение относительно abs_floor
    rad = np.array([155.0, 310.0, 100.0, 0.5])
    assert g.check(rad, base) == (True, ["RM2"])


def test_gate_allows_quiet_channels_budget():
    g = gate(max_quiet_channels=1)
    base = np.array([155.0, 310.0, 100.0, 1.0])
    assert g.check(np.array([140.0, 310.0, 100.0, 1.0]), base)[0] is False
    assert g.check(np.array([140.0, 290.0, 100.0, 1.0]), base)[0] is True


def make_detector():
    rng = np.random.default_rng(0)
    E = rng.normal(0, 1, size=(50, 6))
    E[:, 3:] = np.nan  # у части событий нет поздних признаков
    W = rng.normal(0, 1, size=(300, 4))
    det = fit_novelty(E, np.array(["A"] * 25 + ["B"] * 25), W, np.array(["A"] * 150 + ["B"] * 150), window_stride=1)
    det.episode_threshold, det.window_threshold = 2.0, 3.0
    return det


def test_episode_distance_nan_aware():
    det = make_detector()
    near = np.array([0.0, 0.0, 0.0, np.nan, np.nan, np.nan])
    d, lab = det.episode_distance(near)
    assert d < 1.5 and lab in ("A", "B")
    far = np.array([50.0, 50.0, 50.0, np.nan, np.nan, np.nan])
    assert det.episode_distance(far)[0] > 10
    assert det.episode_distance(np.full(6, np.nan)) == (0.0, None)


def test_assess_flags_novel_and_keeps_known():
    det = make_detector()
    g = AnomalyGate(channel_index=np.arange(4), channel_names=NAMES)
    vals = np.zeros((12, 4))
    w = ObservationWindow("w", "r", 0, 110, NAMES, vals, vals + 1.0)
    ep = EpisodeContext("e", "o", NAMES, np.zeros(4), np.zeros((3, 4)), baseline_raw=np.full(4, 1.0))
    known = det.assess(np.zeros(4), np.array([0, 0, 0, np.nan, np.nan, np.nan]), g, w, ep)
    assert not known.novel and not known.anomaly and known.basis == "episode+window"
    novel = det.assess(np.full(4, 40.0), None, g, w, None)
    assert novel.novel and novel.basis == "window" and novel.anomaly  # без памяти отклонение не проверяется


def test_novelty_save_load(tmp_path):
    det = make_detector()
    det.metadata = {"fingerprint": "t"}
    det.save(tmp_path)
    again = NoveltyDetector.load(tmp_path)
    assert again.episode_threshold == pytest.approx(2.0) and again.version == det.version
    q = np.array([0.5, 0.1, -0.2, np.nan, np.nan, np.nan])
    assert again.episode_distance(q)[0] == pytest.approx(det.episode_distance(q)[0])
