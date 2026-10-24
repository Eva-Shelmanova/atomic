#!/usr/bin/env python3
"""
Обучение и калибровка генератора гипотез (этап B1).

    uv run python scripts/train_generator.py [--kinds logreg hgb]

Окна train/val проходят через информационный барьер; метки берутся из LabelStore
только здесь, в офлайн-скрипте. Кандидаты сравниваются по log-loss на val после
калибровки; выборка test не используется (она для benchmark этапа C).
Результат: data/models/generator/{generator.joblib, generator.json}.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atomic.diagnostics.barrier import InformationBarrier  # noqa: E402
from atomic.diagnostics.config import default_config  # noqa: E402
from atomic.diagnostics.features import batch_features  # noqa: E402
from atomic.diagnostics.generator import (  # noqa: E402
    HypothesisGenerator,
    expected_calibration_error,
    nll,
    topk_accuracy,
)
from atomic.diagnostics.labels import LabelStore  # noqa: E402
from atomic.diagnostics.nppad import NPPADDataset  # noqa: E402


def load_split(ds, labels, barrier, split, classes):
    xs, ys = [], []
    for run_id in ds.split(split):
        y = classes.index(labels.scenario(run_id))
        for w in ds.windows(run_id):
            xs.append(barrier.restrict(w).values)
            ys.append(y)
    return batch_features(xs), np.asarray(ys, dtype=int)


def metrics(gen, X, y, k):
    raw = gen.predict_proba(X, calibrated=False)
    cal = gen.predict_proba(X, calibrated=True)
    return {
        "n_windows": int(len(y)),
        "top1": topk_accuracy(cal, y, 1),
        f"top{k}": topk_accuracy(cal, y, k),
        "nll_raw": nll(raw, y),
        "nll_calibrated": nll(cal, y),
        "ece_raw": expected_calibration_error(raw, y),
        "ece_calibrated": expected_calibration_error(cal, y),
        "temperature": gen.temperature,
    }


def per_class(gen, X, y, classes, k):
    cal = gen.predict_proba(X)
    out = {}
    for i, c in enumerate(classes):
        m = y == i
        out[c] = {"n": int(m.sum()), "top1": topk_accuracy(cal[m], y[m], 1), f"top{k}": topk_accuracy(cal[m], y[m], k)}
    pred = cal.argmax(axis=1)
    confusion = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y, pred, strict=True):
        confusion[t, p] += 1
    return out, confusion.tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", nargs="+", default=["logreg", "hgb"])
    args = ap.parse_args()

    cfg = default_config()
    ds = NPPADDataset()
    labels = LabelStore.load(ds.directory)
    barrier = InformationBarrier()
    classes = tuple(sorted(cfg.benchmark_scenarios))
    k = cfg.top_k

    t0 = time.time()
    X_tr, y_tr = load_split(ds, labels, barrier, "train", classes)
    X_va, y_va = load_split(ds, labels, barrier, "val", classes)
    print(f"Признаки: train {X_tr.shape}, val {X_va.shape} ({time.time() - t0:.1f} с)")

    candidates = {}
    for kind in args.kinds:
        t0 = time.time()
        gen = HypothesisGenerator.fit(X_tr, y_tr, X_va, y_va, barrier.allowed, classes, kind, cfg.seed, k)
        m = metrics(gen, X_va, y_va, k)
        m["fit_seconds"] = round(time.time() - t0, 1)
        candidates[kind] = (gen, m)
        print(f"{kind:7s} val: top1={m['top1']:.3f} top{k}={m[f'top{k}']:.3f} "
              f"NLL {m['nll_raw']:.3f}→{m['nll_calibrated']:.3f} ECE {m['ece_raw']:.3f}→{m['ece_calibrated']:.3f} "
              f"T={m['temperature']:.2f} ({m['fit_seconds']} с)")

    best_kind = min(candidates, key=lambda kd: candidates[kd][1]["nll_calibrated"])
    gen, best = candidates[best_kind]
    pc, confusion = per_class(gen, X_va, y_va, classes, k)
    fp_src = f"{ds.manifest['source']['commit']}|{ds.manifest['config_fingerprint']}|{best_kind}|{cfg.seed}"
    gen.metadata = {
        "fingerprint": hashlib.sha256(fp_src.encode()).hexdigest()[:10],
        "trained_on": "train", "calibrated_on": "val", "seed": cfg.seed,
        "data_source_commit": ds.manifest["source"]["commit"],
        "data_config_fingerprint": ds.manifest["config_fingerprint"],
        "selection": "min calibrated NLL on val",
        "val_metrics": {kd: m for kd, (_, m) in candidates.items()},
        "val_per_class": pc, "val_confusion": confusion,
    }
    out = gen.save(cfg.resolve("models") / "generator")
    print(f"Выбрана модель: {best_kind}; сохранено {out}; версия {gen.version}")
    print(json.dumps(pc, ensure_ascii=False))


if __name__ == "__main__":
    main()
