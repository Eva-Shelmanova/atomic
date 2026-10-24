"""
Загрузчик и нормализатор NPPAD (требование 2 задания).

Две части:
1. build_processed() — офлайн-подготовка: чтение CSV PCTRAN, приведение
   столбцов к реестру каналов, проверка сетки времени, непрозрачные run_id,
   метки в отдельный LabelStore, разбиение по прогонам, статистики
   нормализации только по train. Результат — data/processed/nppad/.
2. NPPADDataset — рабочий доступ к прогонам и окнам. Меток не содержит.

Формат data/processed/nppad/:
    runs.npz       data (N, C) float32 в физических единицах, time (N,),
                   offsets (R+1,), run_ids (R,)
    manifest.json  версии, порядок каналов, нормализация, разбиение, статистика
    labels.json    run_id → сценарий, тяжесть, исходный файл (LabelStore)
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from atomic.contracts.hypothesis import EpisodeContext, ObservationWindow
from atomic.diagnostics.channels import ChannelRegistry, default_registry
from atomic.diagnostics.config import PROJECT_ROOT, DiagnosticsConfig, default_config
from atomic.diagnostics.labels import LabelStore, RunLabel
from atomic.diagnostics.splits import stratified_run_split

DATA_STEP_S = 10.0  # шаг записи PCTRAN в NPPAD
PROCESSED_FORMAT = "nppad-processed/1"

_MALFUNCTION_RE = re.compile(r"Malfunction\s*#\s*(\d+)\s*Fraction\s*=\s*(-?[\d.]+)\s*%")


def make_run_id(scenario: str, stem: str) -> str:
    """Непрозрачный id прогона: путь NPPAD содержит имя сценария и не должен утекать."""
    return hashlib.sha256(f"nppad:{scenario}/{stem}".encode()).hexdigest()[:12]


def make_window_id(run_id: str, start: int) -> str:
    return hashlib.sha256(f"{run_id}:{start}".encode()).hexdigest()[:16]


def _stem_number(stem: str) -> float | None:
    """Номер файла NPPAD = тяжесть в % (у RI имена отрицательные: -1 … -100)."""
    try:
        return float(stem)
    except ValueError:
        return None


def _parse_report(path: Path) -> tuple[int | None, float | None]:
    """Первая строка Malfunction в Transient Report: номер неисправности и доля, %."""
    if not path.exists():
        return None, None
    m = _MALFUNCTION_RE.search(path.read_text(encoding="utf-8", errors="ignore"))
    if not m:
        return None, None
    return int(m.group(1)), float(m.group(2))


@dataclass
class RawRun:
    scenario: str
    stem: str
    time: np.ndarray  # (T,)
    data: np.ndarray  # (T, C) в порядке реестра
    dropped_columns: tuple[str, ...]


def read_raw_run(path: Path, scenario: str, registry: ChannelRegistry) -> RawRun:
    with path.open(newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = [r for r in reader if r]
    col = {name: i for i, name in enumerate(header)}
    if registry.time_column not in col:
        raise ValueError(f"{path}: нет столбца {registry.time_column}")
    missing = [c for c in registry.names if c not in col]
    if missing:
        raise ValueError(f"{path}: нет каналов {missing}")
    extra = tuple(h for h in header if h not in registry.names and h != registry.time_column)
    unexpected = [h for h in extra if h not in registry.ignored_columns]
    if unexpected:
        raise ValueError(f"{path}: неожиданные столбцы {unexpected}")
    arr = np.asarray(rows, dtype=np.float64)
    order = np.argsort(arr[:, col[registry.time_column]], kind="stable")
    arr = arr[order]
    time = arr[:, col[registry.time_column]]
    data = arr[:, [col[c] for c in registry.names]]
    if not np.all(np.isfinite(data)):
        raise ValueError(f"{path}: нечисловые значения")
    return RawRun(scenario=scenario, stem=path.stem, time=time, data=data, dropped_columns=extra)


def _git_commit(path: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:
        return None


def build_processed(
    config: DiagnosticsConfig | None = None,
    registry: ChannelRegistry | None = None,
    raw_dir: Path | None = None,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Подготовка NPPAD одной командой. Возвращает манифест."""
    cfg = config or default_config()
    reg = registry or default_registry()
    raw_dir = raw_dir or cfg.resolve("nppad_raw")
    reports_dir = cfg.resolve("nppad_reports")
    out_dir = out_dir or cfg.resolve("processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    reg.validate_allowed(cfg.allowed_channels)

    scenarios = list(cfg.benchmark_scenarios) + list(cfg.probe_scenarios)
    runs: list[tuple[str, RawRun]] = []
    labels: dict[str, RunLabel] = {}
    issues: list[str] = []
    dropped: dict[str, int] = {}
    for scenario in scenarios:
        sdir = raw_dir / scenario
        if not sdir.is_dir():
            raise FileNotFoundError(f"нет каталога сценария {sdir}")
        files = sorted(sdir.glob("*.csv"), key=lambda p: (_stem_number(p.stem) is None, _stem_number(p.stem) or 0, p.stem))
        for f in files:
            run = read_raw_run(f, scenario, reg)
            steps = np.diff(run.time)
            if len(steps) and not np.allclose(steps, DATA_STEP_S):
                issues.append(f"{scenario}/{f.name}: нерегулярный шаг времени")
            for c in run.dropped_columns:
                dropped[c] = dropped.get(c, 0) + 1
            run_id = make_run_id(scenario, run.stem)
            malfunction, fraction = _parse_report(reports_dir / scenario / f"{run.stem}Transient Report.txt")
            severity = fraction if fraction is not None else _stem_number(run.stem)
            labels[run_id] = RunLabel(
                run_id=run_id,
                scenario=scenario,
                severity=severity,
                source_file=f"{scenario}/{f.name}",
                malfunction=malfunction,
            )
            runs.append((run_id, run))

    runs.sort(key=lambda x: x[0])  # порядок хранения не зависит от сценария
    run_ids = [r for r, _ in runs]
    lengths = [len(run.time) for _, run in runs]
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    data = np.concatenate([run.data for _, run in runs]).astype(np.float32)
    time = np.concatenate([run.time for _, run in runs]).astype(np.float32)

    bench = set(cfg.benchmark_scenarios)
    split = stratified_run_split(
        {r: labels[r].scenario for r in run_ids if labels[r].scenario in bench},
        cfg.split_fractions,
        cfg.seed,
    )
    split["probe"] = sorted(r for r in run_ids if labels[r].scenario not in bench)

    idx = {r: i for i, r in enumerate(run_ids)}
    train_rows = np.concatenate(
        [data[offsets[idx[r]] : offsets[idx[r] + 1]] for r in split["train"]]
    ).astype(np.float64)
    mean = train_rows.mean(axis=0)
    std = train_rows.std(axis=0)
    constant = [reg.names[i] for i in np.where(std < 1e-9)[0]]
    std = np.where(std < 1e-9, 1.0, std)

    np.savez_compressed(out_dir / "runs.npz", data=data, time=time, offsets=offsets, run_ids=np.array(run_ids))
    LabelStore(labels).save(out_dir)

    per_scenario = {
        s: {
            "runs": sum(1 for r in run_ids if labels[r].scenario == s),
            "rows": int(sum(lengths[idx[r]] for r in run_ids if labels[r].scenario == s)),
        }
        for s in scenarios
    }
    manifest = {
        "format": PROCESSED_FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "dataset": "NPPAD (Qi et al., Sci Data 9, 766, 2022)",
            "repository": "https://github.com/thu-inet/NuclearPowerPlantAccidentData",
            "commit": _git_commit(raw_dir),
            "subset": "Operation_csv_data",
        },
        "config_version": cfg.version,
        "config_fingerprint": cfg.data_fingerprint(),
        "channel_schema_version": reg.schema_version,
        "channels": list(reg.names),
        "time_step_s": DATA_STEP_S,
        "n_runs": len(run_ids),
        "n_rows": int(len(time)),
        "per_scenario": per_scenario,
        "dropped_columns": dropped,
        "constant_channels_train": constant,
        "issues": issues,
        "normalization": {
            "method": "z-score",
            "fitted_on": "train",
            "mean": mean.tolist(),
            "std": std.tolist(),
        },
        "split": split,
        "split_seed": cfg.seed,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return manifest


class NPPADDataset:
    """Рабочий доступ к обработанному NPPAD. Метки сценариев здесь недоступны."""

    def __init__(self, directory: Path | None = None, config: DiagnosticsConfig | None = None):
        self.config = config or default_config()
        self.directory = directory or self.config.resolve("processed")
        manifest_path = self.directory / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"{manifest_path} не найден. Подготовьте данные: uv run python scripts/load_nppad.py"
            )
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        npz = np.load(self.directory / "runs.npz")
        self._data = npz["data"]
        self._time = npz["time"]
        self._offsets = npz["offsets"]
        self.run_ids: tuple[str, ...] = tuple(str(r) for r in npz["run_ids"])
        self._index = {r: i for i, r in enumerate(self.run_ids)}
        self.channels: tuple[str, ...] = tuple(self.manifest["channels"])
        norm = self.manifest["normalization"]
        self._mean = np.asarray(norm["mean"], dtype=np.float32)
        self._std = np.asarray(norm["std"], dtype=np.float32)

    def split(self, name: str) -> list[str]:
        return list(self.manifest["split"][name])

    def run_length(self, run_id: str) -> int:
        i = self._index[run_id]
        return int(self._offsets[i + 1] - self._offsets[i])

    def run_raw(self, run_id: str) -> tuple[np.ndarray, np.ndarray]:
        """(time, data) прогона в физических единицах."""
        i = self._index[run_id]
        a, b = self._offsets[i], self._offsets[i + 1]
        return self._time[a:b], self._data[a:b]

    def normalize(self, data: np.ndarray) -> np.ndarray:
        return (data - self._mean) / self._std

    def windows(
        self, run_id: str, length: int | None = None, stride: int | None = None
    ) -> Iterator[ObservationWindow]:
        """Окна прогона. Прогон короче окна даёт ноль окон (учитывается в манифесте оценщика)."""
        length = length or self.config.window_length
        stride = stride or self.config.window_stride
        time, raw = self.run_raw(run_id)
        norm = self.normalize(raw)
        for start in range(0, len(time) - length + 1, stride):
            sl = slice(start, start + length)
            yield ObservationWindow(
                window_id=make_window_id(run_id, start),
                run_id=run_id,
                t_start=float(time[start]),
                t_end=float(time[start + length - 1]),
                channels=self.channels,
                values=norm[sl],
                raw_values=raw[sl],
            )

    def episode(self, run_id: str, upto_t: float, horizon_s: float = 120.0) -> EpisodeContext:
        """Контекст события для окна, заканчивающегося в upto_t: только прошлое (D-030)."""
        time, raw = self.run_raw(run_id)
        norm = self.normalize(raw)
        limit = min(float(upto_t), float(time[0]) + horizon_s)
        n = int(np.searchsorted(time, limit, side="right"))
        return EpisodeContext(
            episode_id=hashlib.sha256(f"episode:{run_id}".encode()).hexdigest()[:12],
            onset_window_id=make_window_id(run_id, 0),
            channels=self.channels,
            baseline=norm[0].copy(),
            early=norm[: max(n, 1)].copy(),
            step_s=DATA_STEP_S,
            baseline_raw=raw[0].astype(np.float64).copy(),
        )

    def iter_windows(self, split: str, **kw: Any) -> Iterator[ObservationWindow]:
        for run_id in self.split(split):
            yield from self.windows(run_id, **kw)

    def resolve_window(self, window_id: str) -> ObservationWindow | None:
        """Поиск окна по id (для REF_RESOLVABLE). Линейный, достаточно для аудита."""
        for run_id in self.run_ids:
            for w in self.windows(run_id):
                if w.window_id == window_id:
                    return w
        return None


def processed_path_hint() -> str:
    return str((PROJECT_ROOT / "data" / "processed" / "nppad").relative_to(PROJECT_ROOT))
