"""Генератор гипотез: формат top-k, калибровка, детерминизм, барьер (этап B1)."""

import numpy as np
import pytest

from atomic.contracts.hypothesis import ObservationWindow, RestrictedWindow
from atomic.diagnostics.features import feature_names, window_features
from atomic.diagnostics.generator import (
    HypothesisGenerator,
    expected_calibration_error,
    fit_temperature,
    nll,
    softmax_temperature,
    topk_accuracy,
)

CH = ("PWNT", "TAVG", "PSGA", "P", "PRB", "LVPZ")
CLASSES = ("A", "B", "C", "D")


def synthetic(n_per_class=60, T=12, seed=0):
    rng = np.random.default_rng(seed)
    X, y, wins = [], [], []
    for ci in range(len(CLASSES)):
        for _ in range(n_per_class):
            w = rng.normal(0, 0.3, size=(T, len(CH)))
            w[:, ci] += np.linspace(0, 2.0, T)  # класс задаёт тренд «своего» канала
            wins.append(w)
            X.append(window_features(w))
            y.append(ci)
    return np.array(X), np.array(y), wins


@pytest.fixture(scope="module")
def gen():
    X, y, _ = synthetic()
    Xv, yv, _ = synthetic(seed=1)
    return HypothesisGenerator.fit(X, y, Xv, yv, CH, CLASSES, "logreg", seed=42, top_k=3)


def test_features_shape_and_slope():
    w = np.zeros((12, 6))
    w[:, 2] = np.arange(12) * 0.5
    f = window_features(w)
    assert f.shape == (42,) and len(feature_names(CH)) == 42
    names = feature_names(CH)
    assert f[names.index("PSGA.slope")] == pytest.approx(0.5)
    assert f[names.index("PSGA.last")] - f[names.index("PSGA.first")] == pytest.approx(5.5)


def test_generate_returns_valid_topk(gen):
    _, _, wins = synthetic(n_per_class=3, seed=2)
    for i, w in enumerate(wins):
        hs = gen.generate(RestrictedWindow(f"w{i}", CH, w))
        assert len(hs.hypotheses) == 3
        assert all(0.0 <= h.confidence <= 1.0 for h in hs.hypotheses)
        assert sum(h.confidence for h in hs.hypotheses) <= 1.0 + 1e-6
        assert [h.rank for h in hs.hypotheses] == [1, 2, 3]
        assert set(hs.scenarios()) <= set(CLASSES)
        assert hs.generator_version.startswith("gen-1.0.0-logreg")


def test_generator_learns_synthetic_classes(gen):
    X, y, _ = synthetic(seed=3)
    assert topk_accuracy(gen.predict_proba(X), y, 1) > 0.9


def test_generator_rejects_unrestricted_window(gen):
    full = ObservationWindow("w", "r", 0, 110, CH + ("WLR",), np.zeros((12, 7)), np.zeros((12, 7)))
    with pytest.raises(ValueError, match="каналы"):
        gen.generate(RestrictedWindow("w", full.channels, full.values))


def test_fit_is_deterministic():
    X, y, _ = synthetic()
    Xv, yv, _ = synthetic(seed=1)
    a = HypothesisGenerator.fit(X, y, Xv, yv, CH, CLASSES, "hgb", seed=7, top_k=3)
    b = HypothesisGenerator.fit(X, y, Xv, yv, CH, CLASSES, "hgb", seed=7, top_k=3)
    assert np.allclose(a.predict_proba(Xv), b.predict_proba(Xv))
    assert a.temperature == pytest.approx(b.temperature)


def test_save_load_roundtrip(gen, tmp_path):
    gen.save(tmp_path)
    again = HypothesisGenerator.load(tmp_path)
    X, _, _ = synthetic(n_per_class=2, seed=4)
    assert np.allclose(gen.predict_proba(X), again.predict_proba(X))
    assert (tmp_path / "generator.json").exists()


def test_temperature_scaling_fixes_overconfidence():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, size=2000)
    correct = rng.random(2000) < 0.6  # реальная точность 60 %
    pred = np.where(correct, y, (y + 1) % 3)
    proba = np.full((2000, 3), 0.01)
    proba[np.arange(2000), pred] = 0.98  # модель уверена на 98 %
    t = fit_temperature(proba, y)
    cal = softmax_temperature(proba, t)
    assert t > 1.0
    assert nll(cal, y) < nll(proba, y)
    assert expected_calibration_error(cal, y) < expected_calibration_error(proba, y)


def test_softmax_temperature_one_is_identity():
    p = np.array([[0.7, 0.2, 0.1]])
    assert np.allclose(softmax_temperature(p, 1.0), p)
