#!/usr/bin/env python3
"""
Подбор параметров верификатора гипотез на val (этапы B2, C).

    uv run python scripts/sweep_verifier.py [--out docs/test-results/<файл>.json]

Генератор применяется к окнам пакетно (один раз), признаки окна и ранней фазы
события кэшируются; затем перебираются параметры. Критерий выбора (D-022):
максимум доли подтверждённых верных гипотез при доле подтверждённых неверных ≤ 10 %;
варианты, уступающие лидеру менее чем на 0,005, считаются равными, и среди них
выбирается вариант с большим top-1 после проверки.
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atomic.contracts.hypothesis import VerdictStatus  # noqa: E402
from atomic.diagnostics.barrier import InformationBarrier  # noqa: E402
from atomic.diagnostics.config import default_config  # noqa: E402
from atomic.diagnostics.features import batch_features  # noqa: E402
from atomic.diagnostics.generator import HypothesisGenerator  # noqa: E402
from atomic.diagnostics.hypothesis_verifier import HypothesisVerifier  # noqa: E402
from atomic.diagnostics.labels import LabelStore  # noqa: E402
from atomic.diagnostics.nppad import NPPADDataset  # noqa: E402
from atomic.diagnostics.signatures import ScenarioSignatures, onset_features  # noqa: E402

MAX_WRONG_CONFIRMED = 0.10
TIE_TOLERANCE = 0.005


def collect(ds, labels, barrier, gen, sig, onset_sig, split, k):
    restricted, feats, onset, truth = [], [], [], []
    for run_id in ds.split(split):
        t = labels.scenario(run_id)
        for w in ds.windows(run_id):
            restricted.append(barrier.restrict(w).values)
            feats.append(sig.features_of(w.values))
            ep = ds.episode(run_id, w.t_end)
            onset.append(onset_features(ep.baseline, ep.early, onset_sig.channel_index))
            truth.append(t)
    proba = gen.predict_proba(batch_features(restricted))
    top = np.argsort(-proba, axis=1)[:, :k]
    hyps = [[gen.classes[j] for j in row] for row in top]
    return hyps, np.array(feats), np.array(onset), truth


def evaluate(verifier, hyps, feats, onset, truth, use_onset):
    tc = tn = wc = wn = top1 = 0
    for h, f, o, t in zip(hyps, feats, onset, truth):
        best, best_key = None, None
        for rank, s in enumerate(h):
            status, ratio, *_ = verifier.verify_one(s, f, "w", h, o if use_onset else None, "o")
            ok = status is VerdictStatus.CONFIRMED
            if s == t:
                tn += 1
                tc += ok
            else:
                wn += 1
                wc += ok
            key = (ratio, -rank)
            if best_key is None or key > best_key:
                best, best_key = s, key
        top1 += best == t
    return {"true_confirmed": tc / max(tn, 1), "wrong_confirmed": wc / max(wn, 1), "top1_after_verification": top1 / len(hyps)}


def probe_rate(verifier, hyps, feats, onset, use_onset):
    return float(np.mean([
        any(verifier.verify_one(s, f, "w", h, o if use_onset else None, "o")[0] is VerdictStatus.CONFIRMED for s in h)
        for h, f, o in zip(hyps, feats, onset)
    ]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = default_config()
    ds = NPPADDataset()
    labels = LabelStore.load(ds.directory)
    gen = HypothesisGenerator.load(cfg.resolve("models") / "generator")
    sig = ScenarioSignatures.load(cfg.resolve("models") / "verifier")
    onset_sig = ScenarioSignatures.load(cfg.resolve("models") / "verifier" / "onset")
    barrier = InformationBarrier()
    k = cfg.top_k
    val = collect(ds, labels, barrier, gen, sig, onset_sig, "val", k)
    probe = collect(ds, labels, barrier, gen, sig, onset_sig, "probe", k)

    rows = []
    grid = [(False, mw, 0, ct) for mw, ct in itertools.product([0.3, 0.5, 0.7], [0.6, 0.7, 0.8, 0.9])]
    grid += [(True, mw, ot, ct) for mw, ot, ct in itertools.product([0.5, 0.7, 0.9], [15, 30], [0.6, 0.7, 0.8, 0.9])]
    for use_onset, mw, ot, ct in grid:
        params = {**cfg.verifier, "min_weight": mw, "top_channels": 15, "onset_top_channels": ot or 15,
                  "confirm_threshold": ct, "reject_threshold": min(cfg.verifier["reject_threshold"], ct - 0.1),
                  "fallback_min_weight": mw, "fallback_confirm_threshold": ct}
        v = HypothesisVerifier(sig, params, onset_signatures=onset_sig if use_onset else None)
        m = evaluate(v, *val, use_onset)
        rows.append({"episode_memory": use_onset, "min_weight": mw, "top_channels": 15, "onset_top_channels": ot,
                     "confirm_threshold": ct, **{k_: round(x, 4) for k_, x in m.items()},
                     "probe_any_confirmed": round(probe_rate(v, probe[0], probe[1], probe[2], use_onset), 4)})
    print(f"{'память':>6} {'min_w':>5} {'onset':>5} {'conf':>5} | {'верн.подтв':>10} {'неверн.подтв':>12} {'top1':>6} {'пробы':>6}")
    for r in rows:
        print(f"{str(r['episode_memory']):>6} {r['min_weight']:5.1f} {r['onset_top_channels']:5d} {r['confirm_threshold']:5.2f} | "
              f"{r['true_confirmed']:10.3f} {r['wrong_confirmed']:12.3f} {r['top1_after_verification']:6.3f} {r['probe_any_confirmed']:6.3f}")
    result = {"criterion": f"max true_confirmed s.t. wrong_confirmed <= {MAX_WRONG_CONFIRMED}; ties < {TIE_TOLERANCE} by top1",
              "rows": rows}
    for mode in (False, True):
        pool = [r for r in rows if r["episode_memory"] == mode and r["wrong_confirmed"] <= MAX_WRONG_CONFIRMED] or \
               [r for r in rows if r["episode_memory"] == mode]
        lead = max(r["true_confirmed"] for r in pool)
        best = max([r for r in pool if lead - r["true_confirmed"] < TIE_TOLERANCE],
                   key=lambda r: (r["top1_after_verification"], r["true_confirmed"], -r["wrong_confirmed"]))
        result["selected_with_memory" if mode else "selected_window_only"] = best
        print("Выбрано", "(с памятью)" if mode else "(только окно)", best)
    if args.out:
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
