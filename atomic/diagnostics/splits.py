"""
Разбиение прогонов на train / val / test.

Разбиение делается по прогонам, а не по окнам, со стратификацией по сценарию
и фиксированным seed. Внутри класса прогоны перемешиваются детерминированно.
"""

from __future__ import annotations

import numpy as np


def stratified_run_split(
    run_scenarios: dict[str, str],
    fractions: tuple[float, float, float],
    seed: int,
) -> dict[str, list[str]]:
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError(f"доли разбиения должны давать 1: {fractions}")
    rng = np.random.default_rng(seed)
    out: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    by_scenario: dict[str, list[str]] = {}
    for run_id, scenario in run_scenarios.items():
        by_scenario.setdefault(scenario, []).append(run_id)
    for scenario in sorted(by_scenario):
        runs = sorted(by_scenario[scenario])
        rng.shuffle(runs)
        n = len(runs)
        n_train = int(round(n * fractions[0]))
        n_val = int(round(n * fractions[1]))
        out["train"].extend(runs[:n_train])
        out["val"].extend(runs[n_train : n_train + n_val])
        out["test"].extend(runs[n_train + n_val :])
    return {k: sorted(v) for k, v in out.items()}
