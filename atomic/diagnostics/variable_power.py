"""
Внешний стресс-тест: сценарии NPPAD с изменением мощности (Variable_Power_Data, D-029).

12 прогонов (4 нормальных изменения мощности 70–100 %, 8 аварий во время изменения
мощности) в обучение не входят. Используются для проверки:
- нормальные изменения мощности не должны диагностироваться как аварии;
- до ввода неисправности — тоже нет; после ввода — ожидается верный сценарий
  или честный исход «новое/не подтверждено» (рабочая точка отличается от обучающей).
Память о событии строится от причинно обнаруженного начала отклонения (D-039, D-042);
оракульный режим (момент ввода неисправности из манифеста) оставлен для сравнения.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from atomic.contracts.hypothesis import EpisodeContext, ObservationWindow
from atomic.diagnostics.channels import ChannelRegistry, default_registry
from atomic.diagnostics.config import PROJECT_ROOT
from atomic.diagnostics.nppad import DATA_STEP_S, NPPADDataset

VP_DIR = PROJECT_ROOT / "NPPAD" / "raw" / "Variable_Power_csv"


@dataclass
class VPRun:
    scenario: str  # например LOCA_80_to_100
    fault_code: str | None  # LOCA / RW / SGATR / SGBTR или None для нормальных
    injection_time_s: float | None
    time: np.ndarray
    raw: np.ndarray  # (T, C) в порядке реестра

    @property
    def run_key(self) -> str:
        return hashlib.sha256(f"vp:{self.scenario}".encode()).hexdigest()[:12]


def load_vp_runs(root: Path = VP_DIR, registry: ChannelRegistry | None = None) -> list[VPRun]:
    reg = registry or default_registry()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    runs = []
    for scen, meta in sorted(manifest.items()):
        with (root / scen / "case1.csv").open() as fh:
            rows = list(csv.reader(fh))
        col = {h: i for i, h in enumerate(rows[0])}
        arr = np.asarray(rows[1:], dtype=np.float64)
        runs.append(VPRun(scen, meta["fault_code"], meta["injection_time_s"], arr[:, col[reg.time_column]],
                          arr[:, [col[c] for c in reg.names]]))
    return runs


def _segments(time: np.ndarray) -> list[tuple[int, int]]:
    """Непрерывные участки с шагом 10 с (в части файлов после точки 0 с есть разрыв)."""
    breaks = np.where(~np.isclose(np.diff(time), DATA_STEP_S))[0]
    starts = [0, *list(breaks + 1)]
    ends = [*list(breaks + 1), len(time)]
    return list(zip(starts, ends, strict=True))


def vp_windows(run: VPRun, ds: NPPADDataset, length: int = 12, stride: int = 6) -> list[ObservationWindow]:
    norm = ds.normalize(run.raw.astype(np.float32))
    out = []
    for a, b in _segments(run.time):
        for st in range(a, b - length + 1, stride):
            sl = slice(st, st + length)
            out.append(ObservationWindow(
                window_id=hashlib.sha256(f"{run.run_key}:{st}".encode()).hexdigest()[:16], run_id=run.run_key,
                t_start=float(run.time[st]), t_end=float(run.time[st + length - 1]), channels=ds.channels,
                values=norm[sl], raw_values=run.raw[sl].astype(np.float32),
            ))
    return out


def segment_start(run: VPRun) -> int:
    """Начало последнего непрерывного участка записи (по нему строятся окна и память о событии)."""
    return _segments(run.time)[-1][0]


def vp_episode(
    run: VPRun, ds: NPPADDataset, upto_t: float, horizon_s: float = 120.0, onsets: np.ndarray | None = None,
) -> EpisodeContext:
    """Память о событии для окна, заканчивающегося в upto_t.

    С onsets (основной режим, D-039/D-042): причинно найденное начало события для каждой точки —
    `DiagnosticPipeline.onset_track(run.raw, segment_start(run))`.
    Без onsets (оракульный режим, только для сравнения): момент ввода неисправности
    из манифеста, для нормальных прогонов — начало участка.
    """
    seg_start = segment_start(run)
    if onsets is not None:
        onset = int(onsets[int(np.searchsorted(run.time, upto_t, side="right")) - 1])
    elif run.injection_time_s is not None and upto_t >= run.injection_time_s:
        onset = int(np.searchsorted(run.time, run.injection_time_s, side="right")) - 1
    else:
        onset = seg_start
    onset = max(onset, seg_start)
    limit = min(upto_t, run.time[onset] + horizon_s)
    end = int(np.searchsorted(run.time, limit, side="right"))
    norm = ds.normalize(run.raw.astype(np.float32))
    return EpisodeContext(
        episode_id=hashlib.sha256(f"episode:{run.run_key}:{onset}".encode()).hexdigest()[:12],
        onset_window_id=hashlib.sha256(f"{run.run_key}:{onset}".encode()).hexdigest()[:16],
        channels=ds.channels, baseline=norm[onset].copy(), early=norm[onset:max(end, onset + 1)].copy(),
        step_s=DATA_STEP_S, baseline_raw=run.raw[onset].copy(),
    )
