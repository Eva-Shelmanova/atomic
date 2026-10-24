"""
Синтетический мини-набор в формате NPPAD для модульных тестов.

Два «сценария» с разной динамикой давления, по 5 прогонов разной тяжести.
Реальные данные не нужны: тесты проверяют логику, а не физику.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from atomic.diagnostics.channels import default_registry
from atomic.diagnostics.config import DiagnosticsConfig, default_config

SYN_SCENARIOS = ("AAA", "BBB")
SYN_PROBE = ("PRB_ONLY",)


def write_synthetic_nppad(root: Path, n_runs: int = 5, length: int = 40) -> Path:
    reg = default_registry()
    raw = root / "raw"
    reports = root / "reports"
    rng = np.random.default_rng(0)
    for scenario in (*SYN_SCENARIOS, *SYN_PROBE):
        (raw / scenario).mkdir(parents=True)
        (reports / scenario).mkdir(parents=True)
        runs = 1 if scenario in SYN_PROBE else n_runs
        for k in range(1, runs + 1):
            t = np.arange(length) * 10.0
            data = rng.normal(100.0, 1.0, size=(length, len(reg.names)))
            p = reg.names.index("P")
            sign = -1.0 if scenario == "AAA" else 1.0
            data[:, p] = 155.0 + sign * k * 0.1 * np.arange(length)
            header = ["TIME", *reg.names]
            # перемешиваем столбцы, как в реальных CSV (порядок ≠ README)
            perm = [0, *list(rng.permutation(len(reg.names)) + 1)]
            arr = np.column_stack([t, data])[:, perm]
            lines = [",".join(header[i] for i in perm)]
            lines += [",".join(f"{v:.6f}" for v in row) for row in arr]
            (raw / scenario / f"{k}.csv").write_text("\n".join(lines) + "\n")
            (reports / scenario / f"{k}Transient Report.txt").write_text(
                f"Reset to IC #1\n000000.5 sec, Malfunction # 7 Fraction = {k:04.1f} %\n"
            )
    return root


def synthetic_config(root: Path) -> DiagnosticsConfig:
    base = default_config()
    raw = copy.deepcopy(base.raw)
    raw["paths"]["nppad_raw"] = str(root / "raw")
    raw["paths"]["nppad_reports"] = str(root / "reports")
    raw["paths"]["processed"] = str(root / "processed")
    raw["scenarios"]["benchmark"] = list(SYN_SCENARIOS)
    raw["scenarios"]["probe_only"] = list(SYN_PROBE)
    raw["scenarios"]["titles"] = {s: s.lower() for s in (*SYN_SCENARIOS, *SYN_PROBE)}
    raw["windows"] = {"length": 8, "stride": 4}
    return DiagnosticsConfig(raw=raw, path=base.path)
