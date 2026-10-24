#!/usr/bin/env python3
"""
Подбор параметров верификатора гипотез на val (этап B2).

    uv run python scripts/sweep_verifier.py [--out docs/test-results/<файл>.json]

Генератор применяется к окнам пакетно (один раз), признаки сигнатур кэшируются;
затем перебираются min_weight, top_channels, confirm_threshold. Критерий выбора
(D-022): максимум доли подтверждённых верных гипотез при доле подтверждённых
неверных ≤ 10 %; варианты, уступающие лучшему менее чем на 0,005, считаются
равными, и среди них выбирается вариант с большим top-1 после проверки.
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
from atomic.diagnostics.signatures import ScenarioSignatures  # noqa: E402

MAX_WRONG_CONFIRMED = 0.10
TIE_TOLERANCE = 0.005


def collect(ds, labels, barrier, gen, sig, split, k):
    restricted, feats, truth = [], [], []
    for run_id in ds.split(split):
        t = labels.scenario(run_id)
        for w in ds.windows(run_id):
            restricted.append(barrier.restrict(w).values)
            feats.append(sig.features_of(w.values))
            truth.append(t)
    proba = gen.predict_proba(batch_features(restricted))
    top = np.argsort(-proba, axis=1)[:, :k]
    hyps = [[gen.classes[j] for j in row] for row in top]
    return hyps, np.array(feats), truth


def evaluate(verifier, hyps, feats, truth):
    tc = tn = wc = wn = top1 = 0
    for h, f, t in zip(hyps, feats, truth):
        best, best_key = None, None
        for rank, s in enumerate(h):
            status, ratio, *_ = verifier.verify_one(s, f, "w", h)
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
        top1 += best == t if t is not None else 0
    return {
        "true_confirmed": tc / max(tn, 1),
        "wrong_confirmed": wc / max(wn, 1),
        "top1_after_verification": top1 / len(hyps) if truth[0] is not None else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = default_config()
    ds = NPPADDataset()
    labels = LabelStore.load(ds.directory)
    gen = HypothesisGenerator.load(cfg.resolve("models") / "generator")
    sig = ScenarioSignatures.load(cfg.resolve("models") / "verifier")
    barrier = InformationBarrier()
    k = cfg.top_k
    hyps, feats, truth = collect(ds, labels, barrier, gen, sig, "val", k)
    p_hyps, p_feats, _ = collect(ds, labels, barrier, gen, sig, "probe", k)

    rows = []
    for mw, tc_, ct in itertools.product([0.3, 0.5, 0.7], [15, 30], [0.6, 0.7, 0.8, 0.9]):
        params = {**cfg.verifier, "min_weight": mw, "top_channels": tc_, "confirm_threshold": ct,
                  "reject_threshold": min(cfg.verifier["reject_threshold"], ct - 0.1)}
        v = HypothesisVerifier(sig, params)
        m = evaluate(v, hyps, feats, truth)
        probe_any = np.mean([
            any(v.verify_one(s, f, "w", h)[0] is VerdictStatus.CONFIRMED for s in h) for h, f in zip(p_hyps, p_feats)
        ])
        rows.append({"min_weight": mw, "top_channels": tc_, "confirm_threshold": ct, **{k_: round(x, 4) for k_, x in m.items()},
                     "probe_any_confirmed": round(float(probe_any), 4)})
    print(f"{'min_w':>5} {'top':>4} {'conf':>5} | {'верн.подтв':>10} {'неверн.подтв':>12} {'top1':>6} {'пробы':>6}")
    for r in rows:
        print(f"{r['min_weight']:5.1f} {r['top_channels']:4d} {r['confirm_threshold']:5.2f} | {r['true_confirmed']:10.3f} "
              f"{r['wrong_confirmed']:12.3f} {r['top1_after_verification']:6.3f} {r['probe_any_confirmed']:6.3f}")
    feasible = [r for r in rows if r["wrong_confirmed"] <= MAX_WRONG_CONFIRMED]
    pool = feasible or rows
    leader = max(r["true_confirmed"] for r in pool)
    ties = [r for r in pool if leader - r["true_confirmed"] < TIE_TOLERANCE]
    best = max(ties, key=lambda r: (r["top1_after_verification"], r["true_confirmed"], -r["wrong_confirmed"]))
    print("Выбрано:", best)
    if args.out:
        Path(args.out).write_text(json.dumps({"criterion": f"max true_confirmed s.t. wrong_confirmed <= {MAX_WRONG_CONFIRMED}; "
                                                           f"ties < {TIE_TOLERANCE} broken by top1_after_verification",
                                              "rows": rows, "selected": best}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
