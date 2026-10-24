"""
Признаки окна для генератора гипотез.

Вход — только значения разрешённых каналов (RestrictedWindow.values, z-нормированные).
На каждый канал 7 признаков: уровень в начале и в конце окна, среднее, σ,
минимум, максимум, наклон линейного тренда (на шаг). Итого 6 × 7 = 42 признака.
"""

from __future__ import annotations

import numpy as np

FEATURES_PER_CHANNEL = ("first", "last", "mean", "std", "min", "max", "slope")
FEATURE_VERSION = "f1"


def feature_names(channels: tuple[str, ...]) -> list[str]:
    return [f"{c}.{f}" for c in channels for f in FEATURES_PER_CHANNEL]


def window_features(values: np.ndarray) -> np.ndarray:
    """values: (T, C) → вектор (C * 7,)."""
    t = np.arange(values.shape[0], dtype=np.float64)
    t_c = t - t.mean()
    denom = float((t_c**2).sum()) or 1.0
    v = values.astype(np.float64)
    slope = (t_c[:, None] * (v - v.mean(axis=0))).sum(axis=0) / denom
    feats = np.stack(
        [v[0], v[-1], v.mean(axis=0), v.std(axis=0), v.min(axis=0), v.max(axis=0), slope], axis=1
    )  # (C, 7)
    return feats.reshape(-1)


def batch_features(windows: list[np.ndarray]) -> np.ndarray:
    return np.stack([window_features(w) for w in windows]) if windows else np.zeros((0, 0))
