#!/usr/bin/env python3
"""
Оценка детектора новизны «исключить один класс» (leave-one-class-out) — этап C, D-032.

    uv run python scripts/eval_novelty_loco.py [--out docs/test-results/<файл>.json]

Для каждого из 11 классов: эталон строится по train без этого класса, порог — 99-й перцентиль
на val без него; доля окон исключённого класса (val), распознанных как новые, — оценка
способности обнаружить действительно неизвестный сценарий. Ложные тревоги — на val известных классов.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atomic.diagnostics.config import default_config  # noqa: E402
from atomic.diagnostics.labels import LabelStore  # noqa: E402
from atomic.diagnostics.novelty import fit_novelty  # noqa: E402
from atomic.diagnostics.nppad import NPPADDataset  # noqa: E402
from atomic.diagnostics.signatures import ScenarioSignatures, onset_features  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = default_config()
    q = float(cfg.raw["novelty"]["threshold_quantile"])
    ds = NPPADDataset()
    labels = LabelStore.load(ds.directory)
    sig = ScenarioSignatures.load(cfg.resolve("models") / "verifier")
    osig = ScenarioSignatures.load(cfg.resolve("models") / "verifier" / "onset")
    E, El, W, Wl = [], [], [], []
    for r in ds.split("train"):
        z = ds.normalize(ds.run_raw(r)[1])
        E.append(onset_features(z[0], z[:13], osig.channel_index)); El.append(labels.scenario(r))  # noqa: E702
        for w in ds.windows(r):
            W.append(sig.features_of(w.values)); Wl.append(labels.scenario(r))  # noqa: E702
    E, El, W, Wl = np.array(E), np.array(El), np.array(W), np.array(Wl)
    VO, VW, Vl = [], [], []
    for r in ds.split("val"):
        for w in ds.windows(r):
            ep = ds.episode(r, w.t_end)
            VO.append(onset_features(ep.baseline, ep.early, osig.channel_index)); VW.append(sig.features_of(w.values))  # noqa: E702
            Vl.append(labels.scenario(r))
    VO, VW, Vl = np.array(VO), np.array(VW), np.array(Vl)
    rows = {}
    for c in sorted(set(El)):
        det = fit_novelty(E[El != c], El[El != c], W[Wl != c], Wl[Wl != c], window_stride=3)
        ep_d = np.array([det.episode_distance(o)[0] for o in VO])
        w_d, _ = det.window_distances(VW)
        known = Vl != c
        te, tw = np.quantile(ep_d[known], q), np.quantile(w_d[known], q)
        flag = (ep_d > te) | (w_d > tw)
        rows[c] = {"detected_as_novel": round(float(flag[~known].mean()), 3),
                   "by_episode": round(float((ep_d[~known] > te).mean()), 3),
                   "by_window": round(float((w_d[~known] > tw).mean()), 3),
                   "false_alarm_known": round(float(flag[known].mean()), 3)}
        print(c, rows[c])
    mean = {k: round(float(np.mean([v[k] for v in rows.values()])), 3) for k in next(iter(rows.values()))}
    print("Среднее:", mean)
    if args.out:
        Path(args.out).write_text(json.dumps({"quantile": q, "per_class": rows, "mean": mean}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
