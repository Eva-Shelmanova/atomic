#!/usr/bin/env python3
"""
Построение детектора новизны и калибровка порогов (этап C, D-032).

    uv run python scripts/train_novelty.py

Эталон — train-события и train-окна 12 benchmark-классов; пороги — перцентиль
расстояний на val (с причинной памятью о событии: у раннего окна доступна только
часть ранней фазы). Результат: data/models/novelty/{novelty.npz, novelty.json}.
"""

import hashlib
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atomic.diagnostics.config import default_config  # noqa: E402
from atomic.diagnostics.labels import LabelStore  # noqa: E402
from atomic.diagnostics.novelty import fit_novelty  # noqa: E402
from atomic.diagnostics.nppad import NPPADDataset  # noqa: E402
from atomic.diagnostics.signatures import ScenarioSignatures, onset_features  # noqa: E402


def main() -> None:
    t0 = time.time()
    cfg = default_config()
    nov = cfg.raw["novelty"]
    ds = NPPADDataset()
    labels = LabelStore.load(ds.directory)
    sig = ScenarioSignatures.load(cfg.resolve("models") / "verifier")
    osig = ScenarioSignatures.load(cfg.resolve("models") / "verifier" / "onset")

    E, El, W, Wl = [], [], [], []
    for r in ds.split("train"):
        _, raw = ds.run_raw(r)
        z = ds.normalize(raw)
        E.append(onset_features(z[0], z[:13], osig.channel_index))
        El.append(labels.scenario(r))
        for w in ds.windows(r):
            W.append(sig.features_of(w.values))
            Wl.append(labels.scenario(r))
    det = fit_novelty(np.array(E), np.array(El), np.array(W), np.array(Wl), window_stride=int(nov["window_stride"]))

    ve, vw = [], []
    for r in ds.split("val"):
        for w in ds.windows(r):
            ep = ds.episode(r, w.t_end)
            ve.append(det.episode_distance(onset_features(ep.baseline, ep.early, osig.channel_index))[0])
            vw.append(sig.features_of(w.values))
    vw_d, _ = det.window_distances(np.array(vw))
    q = float(nov["threshold_quantile"])
    det.episode_threshold = float(np.quantile(ve, q))
    det.window_threshold = float(np.quantile(vw_d, q))
    far = float(np.mean((np.array(ve) > det.episode_threshold) | (vw_d > det.window_threshold)))
    fp = f"{ds.manifest['source']['commit']}|{ds.manifest['config_fingerprint']}|{cfg.fingerprint('novelty')}"
    det.metadata = {
        "fingerprint": hashlib.sha256(fp.encode()).hexdigest()[:10], "reference": "train", "calibrated_on": "val",
        "threshold_quantile": q, "val_false_alarm_rate": round(far, 4), "episode_ref_runs": len(E),
        "window_ref_windows": int(len(det.window_ref)), "data_source_commit": ds.manifest["source"]["commit"],
    }
    det.save(cfg.resolve("models") / "novelty")
    print(f"Эталон: {len(E)} событий, {len(det.window_ref)} окон; пороги: событие {det.episode_threshold:.3f}, окно {det.window_threshold:.3f}")
    print(f"Ложные тревоги на val (известные сценарии): {far:.3f}; версия {det.version}; {time.time() - t0:.1f} с")


if __name__ == "__main__":
    main()
