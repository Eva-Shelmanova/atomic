#!/usr/bin/env python3
"""
Построение сигнатур сценариев для верификатора гипотез (этап B2).

    uv run python scripts/train_verifier.py

Только train-прогоны benchmark-классов. Каналы: наблюдаемые (без label_proxy, D-007)
и не постоянные на train. Результат: data/models/verifier/{signatures.npz, signatures.json}.
"""

import hashlib
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atomic.diagnostics.channels import default_registry  # noqa: E402
from atomic.diagnostics.config import default_config  # noqa: E402
from atomic.diagnostics.labels import LabelStore  # noqa: E402
from atomic.diagnostics.nppad import NPPADDataset  # noqa: E402
from atomic.diagnostics.signatures import (  # noqa: E402
    ONSET_FEATURES,
    ScenarioSignatures,
    onset_features,
    run_window_features,
)


def main() -> None:
    t0 = time.time()
    cfg = default_config()
    reg = default_registry()
    ds = NPPADDataset()
    labels = LabelStore.load(ds.directory)
    constant = set(ds.manifest["constant_channels_train"])
    excluded = (reg.label_proxies() if cfg.verifier.get("exclude_label_proxy", True) else frozenset()) | constant
    channels = tuple(c for c in ds.channels if c not in excluded)
    idx = np.array([ds.channels.index(c) for c in channels])

    feats: dict[str, list[np.ndarray]] = {s: [] for s in cfg.benchmark_scenarios}
    onset: dict[str, list[np.ndarray]] = {s: [] for s in cfg.benchmark_scenarios}
    for run_id in ds.split("train"):
        _, raw = ds.run_raw(run_id)
        full = ds.normalize(raw)
        norm = full[:, idx]
        f = run_window_features(norm, cfg.window_length, cfg.window_stride)
        if len(f):
            feats[labels.scenario(run_id)].append(f)
        # ранняя фаза: один вектор на прогон (исходное состояние — точка 0 с)
        onset[labels.scenario(run_id)].append(onset_features(full[0], full[:13], idx)[None, :])
    by_scenario = {s: np.concatenate(v) for s, v in feats.items()}
    onset_by_scenario = {s: np.concatenate(v) for s, v in onset.items()}

    fp_src = f"{ds.manifest['source']['commit']}|{ds.manifest['config_fingerprint']}|{','.join(channels)}"
    sig = ScenarioSignatures.build(
        by_scenario, channels, idx,
        metadata={
            "fingerprint": hashlib.sha256(fp_src.encode()).hexdigest()[:10],
            "built_on": "train",
            "excluded_label_proxy": sorted(reg.label_proxies() & excluded),
            "excluded_constant": sorted(constant),
            "data_source_commit": ds.manifest["source"]["commit"],
            "data_config_fingerprint": ds.manifest["config_fingerprint"],
        },
    )
    out = sig.save(cfg.resolve("models") / "verifier")
    onset_sig = ScenarioSignatures.build(
        onset_by_scenario, channels, idx, metadata={**sig.metadata, "family": "onset", "built_on": "train (1 вектор на прогон)"},
        feature_labels=ONSET_FEATURES,
    )
    onset_sig.save(cfg.resolve("models") / "verifier" / "onset")
    print(f"Сигнатуры ранней фазы: {len(channels) * len(ONSET_FEATURES)} признаков, прогонов на сценарий "
          f"{ {s: int(c) for s, c in zip(onset_sig.scenarios, onset_sig.counts)} }")
    print(f"Каналов в сигнатурах: {len(channels)} (исключено label_proxy {len(reg.label_proxies() & excluded)}, "
          f"постоянных {len(constant)}); признаков: {len(channels) * 2}")
    print("Окон на сценарий:", {s: int(c) for s, c in zip(sig.scenarios, sig.counts)})
    print(f"Сохранено {out}; версия {sig.version}; {time.time() - t0:.1f} с")


if __name__ == "__main__":
    main()
