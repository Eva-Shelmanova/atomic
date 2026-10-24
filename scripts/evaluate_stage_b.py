#!/usr/bin/env python3
"""
Промежуточная оценка этапа B на val-прогонах (и пробах вне распределения).

    uv run python scripts/evaluate_stage_b.py [--out docs/test-results/<файл>.json]

Это не итоговый benchmark: val использовалась для выбора модели и калибровки.
Выборка test не трогается до этапа C.
"""

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atomic.contracts.hypothesis import VerdictStatus  # noqa: E402
from atomic.diagnostics.analogues import AnalogueFinder  # noqa: E402
from atomic.diagnostics.barrier import InformationBarrier  # noqa: E402
from atomic.diagnostics.config import default_config  # noqa: E402
from atomic.diagnostics.generator import HypothesisGenerator  # noqa: E402
from atomic.diagnostics.hypothesis_verifier import HypothesisVerifier  # noqa: E402
from atomic.diagnostics.labels import LabelStore  # noqa: E402
from atomic.diagnostics.nppad import NPPADDataset  # noqa: E402
from atomic.diagnostics.signatures import ScenarioSignatures  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.split == "test":
        sys.exit("test зарезервирован для benchmark этапа C")

    cfg = default_config()
    ds = NPPADDataset()
    labels = LabelStore.load(ds.directory)
    barrier = InformationBarrier()
    gen = HypothesisGenerator.load(cfg.resolve("models") / "generator")
    finder = AnalogueFinder.load()
    verifier = HypothesisVerifier(ScenarioSignatures.load(cfg.resolve("models") / "verifier"), cfg.verifier, finder)
    k = cfg.top_k

    stats = Counter()
    true_status, wrong_status = Counter(), Counter()
    per_class = defaultdict(Counter)
    t_gen = t_ver = 0.0
    n = 0
    for run_id in ds.split(args.split):
        truth = labels.scenario(run_id)
        for w in ds.windows(run_id):
            n += 1
            t0 = time.perf_counter()
            hs = gen.generate(barrier.restrict(w))
            t1 = time.perf_counter()
            verdicts = verifier.verify(w, hs, with_analogues=False)
            t2 = time.perf_counter()
            t_gen += t1 - t0
            t_ver += t2 - t1
            scen = hs.scenarios()
            stats["gen_top1"] += scen[0] == truth
            stats["gen_topk"] += truth in scen
            for v in verdicts:
                (true_status if v.hypothesis.scenario == truth else wrong_status)[v.status.value] += 1
                stats["evidence"] += len(v.evidence_for) + len(v.evidence_against)
                stats["evidence_ref_ok"] += sum(
                    1 for e in (*v.evidence_for, *v.evidence_against) if e.window_id == w.window_id and e.channel in ds.channels
                )
            # повторное ранжирование: доля подтверждений, при равенстве — уверенность генератора
            best = max(verdicts, key=lambda v: (v.support_ratio, v.hypothesis.confidence))
            stats["verified_top1"] += best.hypothesis.scenario == truth
            top = verdicts[0]
            stats["top1_confirmed_correct"] += top.status is VerdictStatus.CONFIRMED and top.hypothesis.scenario == truth
            stats["top1_confirmed_wrong"] += top.status is VerdictStatus.CONFIRMED and top.hypothesis.scenario != truth
            pc = per_class[truth]
            pc["n"] += 1
            pc["gen_top1"] += scen[0] == truth
            pc["verified_top1"] += best.hypothesis.scenario == truth
            pc["true_confirmed"] += any(v.status is VerdictStatus.CONFIRMED and v.hypothesis.scenario == truth for v in verdicts)

    probe = Counter()
    for run_id in ds.split("probe"):
        for w in ds.windows(run_id):
            verdicts = verifier.verify(w, gen.generate(barrier.restrict(w)), with_analogues=False)
            probe["windows"] += 1
            probe["any_confirmed"] += any(v.status is VerdictStatus.CONFIRMED for v in verdicts)
            probe["top1_confirmed"] += verdicts[0].status is VerdictStatus.CONFIRMED

    t0 = time.perf_counter()
    for s in cfg.benchmark_scenarios:
        finder.find(s, k=cfg.verifier.get("analogues_k", 3))
    t_an = (time.perf_counter() - t0) / len(cfg.benchmark_scenarios)

    def share(c: Counter) -> dict:
        tot = sum(c.values()) or 1
        return {key: round(c[key] / tot, 4) for key in ("confirmed", "uncertain", "rejected")}

    result = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split": args.split,
        "windows": n,
        "generator_version": gen.version,
        "verifier_version": verifier.version,
        "verifier_params": {k_: cfg.verifier[k_] for k_ in ("min_weight", "top_channels", "confirm_threshold", "reject_threshold")},
        "generator": {"top1": round(stats["gen_top1"] / n, 4), f"top{k}": round(stats["gen_topk"] / n, 4)},
        "verifier": {
            "true_hypotheses": share(true_status),
            "wrong_hypotheses": share(wrong_status),
            "top1_after_verification": round(stats["verified_top1"] / n, 4),
            "top1_confirmed_and_correct": round(stats["top1_confirmed_correct"] / n, 4),
            "top1_confirmed_but_wrong": round(stats["top1_confirmed_wrong"] / n, 4),
            "evidence_with_channel_window_ref": round(stats["evidence_ref_ok"] / max(stats["evidence"], 1), 4),
            "evidence_per_hypothesis": round(stats["evidence"] / (n * k), 2),
        },
        "probes_out_of_distribution": {
            "windows": probe["windows"],
            "any_hypothesis_confirmed": round(probe["any_confirmed"] / max(probe["windows"], 1), 4),
            "top1_confirmed": round(probe["top1_confirmed"] / max(probe["windows"], 1), 4),
        },
        "latency_ms_per_window": {
            "generator": round(1000 * t_gen / n, 3),
            "hypothesis_verifier": round(1000 * t_ver / n, 3),
            "analogues_per_scenario": round(1000 * t_an, 3),
        },
        "per_class": {
            c: {"n": v["n"], "gen_top1": round(v["gen_top1"] / v["n"], 3),
                "verified_top1": round(v["verified_top1"] / v["n"], 3),
                "true_confirmed": round(v["true_confirmed"] / v["n"], 3)}
            for c, v in sorted(per_class.items())
        },
    }
    text = json.dumps(result, ensure_ascii=False, indent=1)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
