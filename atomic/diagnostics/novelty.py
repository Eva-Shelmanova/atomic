"""
Проверка отклонения и новизны (этап C, D-032).

Два вопроса, на которые верификатор гипотез сам ответить не может:
1. Есть ли вообще отклонение от исходного состояния? (AnomalyGate)
   Число наблюдаемых каналов, у которых значение в конце окна отличается от
   исходного состояния больше чем на rel_threshold (по умолчанию 1 %). Порог по
   числу каналов задан по прогонам без события (ATWS и SP в NPPAD: неисправность
   с долей 0 %, станция остаётся в стационаре; дрейфуют 4 канала).
2. Похоже ли событие на известные сценарии? (NoveltyDetector)
   - по ранней фазе события: RMS-расстояние (робастная нормировка) до ближайшего
     train-события в пространстве признаков d30/d60/d120, только по уже наблюдённым признакам;
   - по текущему окну: евклидово расстояние до ближайшего train-окна (уровень и тренд).
   Порог каждого — 99-й перцентиль на val (суммарно ≈ 2 % ложных тревог на известных
   сценариях). Событие новое, если превышен любой из порогов.

Метки train используются только для подписи «ближайший известный сценарий».
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from atomic.contracts.hypothesis import EpisodeContext, NoveltyAssessment, ObservationWindow

NOVELTY_SCHEMA = "1.0.0"


@dataclass
class AnomalyGate:
    channel_index: np.ndarray  # наблюдаемые каналы (индексы в датасете)
    channel_names: tuple[str, ...]
    rel_threshold: float = 0.01
    max_quiet_channels: int = 0  # столько отклонившихся каналов допускается без события
    abs_floor: float = 1e-3  # нижняя граница |исходного значения| для относительного отклонения
    drift_channels: tuple[str, ...] = ()  # дрейфуют и без события — в подсчёт не входят

    def deviating(self, current_raw: np.ndarray, baseline_raw: np.ndarray) -> list[str]:
        cur = current_raw[self.channel_index].astype(np.float64)
        base = baseline_raw[self.channel_index].astype(np.float64)
        rel = np.abs(cur - base) / np.maximum(np.abs(base), self.abs_floor)
        return [self.channel_names[i] for i in np.where(rel > self.rel_threshold)[0]
                if self.channel_names[i] not in self.drift_channels]

    def check(self, current_raw: np.ndarray, baseline_raw: np.ndarray) -> tuple[bool, list[str]]:
        dev = self.deviating(current_raw, baseline_raw)
        return len(dev) > self.max_quiet_channels, dev


@dataclass
class NoveltyDetector:
    episode_ref: np.ndarray  # (R, Fe) признаки ранней фазы train-событий
    episode_labels: np.ndarray  # (R,)
    episode_center: np.ndarray
    episode_scale: np.ndarray
    window_ref: np.ndarray  # (N, Fw) признаки train-окон (стандартизованные)
    window_labels: np.ndarray
    window_mean: np.ndarray
    window_std: np.ndarray
    episode_threshold: float = np.inf
    window_threshold: float = np.inf
    metadata: dict | None = None

    @property
    def version(self) -> str:
        return f"nov-{NOVELTY_SCHEMA}-{(self.metadata or {}).get('fingerprint', 'dev')}"

    # --- расстояния ---
    def episode_distance(self, onset: np.ndarray) -> tuple[float, str | None]:
        q = (onset - self.episode_center) / self.episode_scale
        m = ~np.isnan(q)
        if not m.any():
            return 0.0, None
        ref = (self.episode_ref[:, m] - self.episode_center[m]) / self.episode_scale[m]
        d = ref - q[m]
        valid = ~np.isnan(d)
        dist = np.sqrt((np.where(valid, d, 0.0) ** 2).sum(1) / np.maximum(valid.sum(1), 1))
        j = int(np.argmin(dist))
        return float(dist[j]), str(self.episode_labels[j])

    def window_distances(self, feats: np.ndarray, batch: int = 2048) -> tuple[np.ndarray, np.ndarray]:
        """Пакетно: (N, Fw) → (расстояния, индексы ближайших)."""
        Q = ((feats - self.window_mean) / self.window_std).astype(np.float32)
        R = self.window_ref.astype(np.float32)
        r2 = (R**2).sum(1)
        dists, idx = np.empty(len(Q)), np.empty(len(Q), dtype=int)
        for a in range(0, len(Q), batch):
            q = Q[a : a + batch]
            d2 = (q**2).sum(1)[:, None] + r2[None, :] - 2 * q @ R.T
            j = d2.argmin(1)
            idx[a : a + batch] = j
            dists[a : a + batch] = np.sqrt(np.maximum(d2[np.arange(len(q)), j], 0))
        return dists, idx

    def window_distance(self, feats: np.ndarray) -> tuple[float, str]:
        d, j = self.window_distances(feats[None, :])
        return float(d[0]), str(self.window_labels[j[0]])

    # --- оценка ---
    def assess(
        self, window_feats: np.ndarray, onset: np.ndarray | None, gate: AnomalyGate,
        window: ObservationWindow, episode: EpisodeContext | None,
    ) -> NoveltyAssessment:
        anomaly, dev = (True, [])
        if episode is not None and episode.baseline_raw is not None:
            anomaly, dev = gate.check(window.raw_values[-1], episode.baseline_raw)
        w_score, w_near = self.window_distance(window_feats)
        e_score, e_near = (None, None)
        if onset is not None:
            e_score, e_near = self.episode_distance(onset)
        novel = w_score > self.window_threshold or (e_score is not None and e_score > self.episode_threshold)
        return NoveltyAssessment(
            anomaly=anomaly, deviating_channels=tuple(dev), novel=bool(novel),
            episode_score=None if e_score is None else round(e_score, 4), episode_threshold=round(self.episode_threshold, 4),
            window_score=round(w_score, 4), window_threshold=round(self.window_threshold, 4),
            nearest_scenario=e_near or w_near, basis="episode+window" if onset is not None else "window",
            detector_version=self.version,
        )

    # --- сохранение ---
    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            directory / "novelty.npz", episode_ref=self.episode_ref, episode_labels=self.episode_labels,
            episode_center=self.episode_center, episode_scale=self.episode_scale, window_ref=self.window_ref,
            window_labels=self.window_labels, window_mean=self.window_mean, window_std=self.window_std,
        )
        meta = {**(self.metadata or {}), "version": self.version, "episode_threshold": self.episode_threshold,
                "window_threshold": self.window_threshold}
        (directory / "novelty.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> NoveltyDetector:
        npz = np.load(directory / "novelty.npz")
        meta = json.loads((directory / "novelty.json").read_text(encoding="utf-8"))
        return cls(
            episode_ref=npz["episode_ref"], episode_labels=npz["episode_labels"], episode_center=npz["episode_center"],
            episode_scale=npz["episode_scale"], window_ref=npz["window_ref"], window_labels=npz["window_labels"],
            window_mean=npz["window_mean"], window_std=npz["window_std"],
            episode_threshold=float(meta["episode_threshold"]), window_threshold=float(meta["window_threshold"]), metadata=meta,
        )


def fit_novelty(
    episode_feats: np.ndarray, episode_labels: np.ndarray, window_feats: np.ndarray, window_labels: np.ndarray,
    window_stride: int = 3, metadata: dict | None = None,
) -> NoveltyDetector:
    import warnings

    with warnings.catch_warnings():  # признаки, не наблюдавшиеся ни в одном событии, ожидаемо пусты
        warnings.simplefilter("ignore", RuntimeWarning)
        center = np.nanmedian(episode_feats, 0)
        iqr = np.nanpercentile(episode_feats, 75, 0) - np.nanpercentile(episode_feats, 25, 0)
    scale = np.where(np.isnan(iqr) | (iqr < 1e-6), 1.0, iqr)
    center = np.where(np.isnan(center), 0.0, center)
    mean, std = window_feats.mean(0), window_feats.std(0) + 1e-6
    ref = ((window_feats - mean) / std)[::window_stride]
    return NoveltyDetector(
        episode_ref=episode_feats, episode_labels=np.asarray(episode_labels), episode_center=center, episode_scale=scale,
        window_ref=ref, window_labels=np.asarray(window_labels)[::window_stride], window_mean=mean, window_std=std,
        metadata=metadata or {},
    )
