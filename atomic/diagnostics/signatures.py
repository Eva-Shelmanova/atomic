"""
Сигнатуры сценариев для верификатора гипотез (этап B2).

Для каждого сценария и каждого наблюдаемого канала по train-окнам считаются
квантили двух признаков окна: уровня (среднее z-значение) и тренда (наклон на шаг).
Квантили q01, q05, q50, q95, q99 задают «типичный» [q05; q95] и «допустимый»
[q01; q99] диапазоны.

Матрица специфичности spec[s, o, p] — доля окон сценария o, у которых признак p
выходит за типичный диапазон сценария s. Чем она выше, тем лучше признак p
отличает s от o. Верификатор использует её как вес свидетельства.

Каналы label_proxy и каналы, постоянные на train, в сигнатуры не входят (D-007).

Два семейства сигнатур (D-030):
- оконные (level, slope) — по текущему окну;
- ранней фазы (d30, d60, d120) — изменение каждого канала через 30, 60 и 120 с
  после начала события относительно исходного состояния. Строятся по одному
  вектору на train-прогон; отсутствующие (прогон короче) значения — NaN.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

QUANTILES = (0.01, 0.05, 0.50, 0.95, 0.99)
FEATURES = ("level", "slope")
ONSET_OFFSETS_S = (30, 60, 120)
ONSET_FEATURES = tuple(f"d{s}" for s in ONSET_OFFSETS_S)
SIGNATURES_SCHEMA = "1.0.0"


def window_matrix_features(values: np.ndarray) -> np.ndarray:
    """values: (N, T, C) → (N, C * 2): для каждого канала уровень и наклон."""
    T = values.shape[1]
    t = np.arange(T, dtype=np.float64) - (T - 1) / 2.0
    denom = float((t**2).sum()) or 1.0
    level = values.mean(axis=1)
    slope = np.einsum("t,ntc->nc", t, values - level[:, None, :]) / denom
    return np.stack([level, slope], axis=2).reshape(values.shape[0], -1)


def run_window_features(norm: np.ndarray, length: int, stride: int) -> np.ndarray:
    """Признаки всех окон одного прогона (norm: (T, C)) без построения ObservationWindow."""
    if len(norm) < length:
        return np.zeros((0, norm.shape[1] * 2))
    view = np.lib.stride_tricks.sliding_window_view(norm, length, axis=0)[::stride]  # (N, C, L)
    return window_matrix_features(np.transpose(view, (0, 2, 1)).astype(np.float64))


def onset_features(baseline: np.ndarray, early: np.ndarray, channel_index: np.ndarray, step_s: float = 10.0) -> np.ndarray:
    """(C_all,), (n, C_all) → (len(channel_index) * 3,): Δ через 30/60/120 с; NaN, если ещё не наблюдалось."""
    out = np.full((len(channel_index), len(ONSET_OFFSETS_S)), np.nan)
    for j, off in enumerate(ONSET_OFFSETS_S):
        k = int(round(off / step_s))
        if k < len(early):
            out[:, j] = early[k, channel_index] - baseline[channel_index]
    return out.reshape(-1)


@dataclass
class ScenarioSignatures:
    scenarios: tuple[str, ...]
    channels: tuple[str, ...]  # каналы, участвующие в сигнатурах
    channel_index: np.ndarray  # индексы этих каналов в порядке датасета
    quantiles: np.ndarray  # (S, P, 5), P = len(channels) * 2
    specificity: np.ndarray  # (S, S, P)
    counts: np.ndarray  # (S,) число train-окон (или прогонов для сигнатур ранней фазы)
    metadata: dict
    feature_labels: tuple[str, ...] = FEATURES

    @property
    def pairs(self) -> list[tuple[str, str]]:
        return [(c, f) for c in self.channels for f in self.feature_labels]

    @property
    def version(self) -> str:
        return f"sig-{SIGNATURES_SCHEMA}-{self.metadata.get('fingerprint', 'dev')}"

    def features_of(self, window_values: np.ndarray) -> np.ndarray:
        """Признаки окна (T, C_all) → (P,) по каналам сигнатуры."""
        v = window_values[:, self.channel_index].astype(np.float64)[None]
        return window_matrix_features(v)[0]

    @classmethod
    def build(
        cls,
        features_by_scenario: dict[str, np.ndarray],
        channels: tuple[str, ...],
        channel_index: np.ndarray,
        metadata: dict | None = None,
        feature_labels: tuple[str, ...] = FEATURES,
    ) -> ScenarioSignatures:
        scenarios = tuple(sorted(features_by_scenario))
        S, P = len(scenarios), len(channels) * len(feature_labels)
        q = np.full((S, P, len(QUANTILES)), np.nan)
        for i, s in enumerate(scenarios):
            F = features_by_scenario[s]
            ok = ~np.all(np.isnan(F), axis=0)
            if ok.any():
                q[i][ok] = np.nanquantile(F[:, ok], QUANTILES, axis=0).T
        spec = np.zeros((S, S, P))
        with np.errstate(invalid="ignore"):
            for i in range(S):
                lo, hi = q[i, :, 1], q[i, :, 3]
                for j, o in enumerate(scenarios):
                    if i == j:
                        continue
                    F = features_by_scenario[o]
                    valid = ~np.isnan(F) & ~np.isnan(lo) & ~np.isnan(hi)
                    out = ((F < lo) | (F > hi)) & valid
                    n = valid.sum(axis=0)
                    spec[i, j] = np.where(n > 0, out.sum(axis=0) / np.maximum(n, 1), 0.0)
        counts = np.array([len(features_by_scenario[s]) for s in scenarios])
        return cls(scenarios, tuple(channels), np.asarray(channel_index), q, spec, counts, metadata or {}, tuple(feature_labels))

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "signatures.npz"
        np.savez_compressed(
            path, quantiles=self.quantiles, specificity=self.specificity, counts=self.counts,
            channel_index=self.channel_index,
        )
        meta = {**self.metadata, "version": self.version, "scenarios": list(self.scenarios),
                "channels": list(self.channels), "features": list(self.feature_labels), "quantiles": list(QUANTILES)}
        (directory / "signatures.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        return path

    @classmethod
    def load(cls, directory: Path) -> ScenarioSignatures:
        meta = json.loads((directory / "signatures.json").read_text(encoding="utf-8"))
        npz = np.load(directory / "signatures.npz")
        return cls(
            scenarios=tuple(meta["scenarios"]), channels=tuple(meta["channels"]),
            channel_index=npz["channel_index"], quantiles=npz["quantiles"],
            specificity=npz["specificity"], counts=npz["counts"], metadata=meta,
            feature_labels=tuple(meta.get("features", FEATURES)),
        )
