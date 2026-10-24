#!/usr/bin/env python3
"""
Офлайн-оценщик (benchmark) диагностического контура — этап C, требования 11–13.

    uv run python scripts/benchmark.py --out docs/test-results/<дата>-benchmark-test.json [--repeat]

Единственный компонент, который видит истинные метки сценариев NPPAD. Оценка — на test-прогонах,
не использованных ни для обучения, ни для настройки (D-025). Разделы:
  generator      — top-1/top-k, калибровка (ECE, NLL);
  verifier       — подтверждение верных/неверных гипотез с памятью о событии и без неё;
  outcomes       — итоговые исходы, точность исхода IDENTIFIED, покрытие, по классам;
  probes         — пробы без события (ожидается NO_ANOMALY) и вне обучаемого множества (ожидается NOVEL);
  variable_power — внешний стресс-тест (изменение мощности);
  answer_verifier— распределение ALLOW/REQUIRE_HUMAN/BLOCK и контрольные испорченные ответы;
  analogues      — precision@3 аналогов по размеченному типу исходного события ETHZ;
  latency        — задержка этапов на окно (p50/p95);
  versions       — паспорт воспроизводимости.
--repeat запускает оценку дважды и сравнивает все детерминированные числа (требование 13).
"""

import argparse
import copy
import hashlib
import json
import platform
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sklearn  # noqa: E402

from atomic.agents.analyst import AnalystAgent  # noqa: E402
from atomic.contracts.hypothesis import DiagnosisOutcome, VerdictStatus  # noqa: E402
from atomic.contracts.verification import EvidenceKind, PolicyEffect  # noqa: E402
from atomic.diagnostics.config import default_config  # noqa: E402
from atomic.diagnostics.features import batch_features  # noqa: E402
from atomic.diagnostics.generator import expected_calibration_error, nll, topk_accuracy  # noqa: E402
from atomic.diagnostics.labels import LabelStore  # noqa: E402
from atomic.diagnostics.nppad import NPPADDataset  # noqa: E402
from atomic.diagnostics.pipeline import DiagnosticPipeline  # noqa: E402
from atomic.diagnostics.telemetry import TelemetryReplay  # noqa: E402
from atomic.diagnostics.variable_power import VPRun, load_vp_runs, stationary_points, vp_episode, vp_windows  # noqa: E402
from atomic.verification import AnswerVerifier, EvidenceResolver  # noqa: E402

R = lambda x: round(float(x), 4)  # noqa: E731


def share(counter: Counter, total: int) -> dict:
    return {k: R(v / max(total, 1)) for k, v in sorted(counter.items())}


def collect(ds, runs, memory=True):
    W, E, T = [], [], []
    for r in runs:
        for w in ds.windows(r):
            W.append(w)
            E.append(ds.episode(r, w.t_end) if memory else None)
            T.append(r)
    return W, E, T


def evaluate(cfg, ds, labels, pipe, answer_sample_every: int) -> dict:
    classes = list(pipe.generator.classes)
    test_runs = ds.split("test")
    W, E, runs = collect(ds, test_runs)
    truth = [labels.scenario(r) for r in runs]
    y = np.array([classes.index(t) for t in truth])

    # --- генератор ---
    X = batch_features([pipe.barrier.restrict(w).values for w in W])
    raw = pipe.generator.predict_proba(X, calibrated=False)
    cal = pipe.generator.predict_proba(X)
    k = pipe.generator.top_k
    res: dict = {"test_windows": len(W), "test_runs": len(test_runs)}
    res["generator"] = {"top1": R(topk_accuracy(cal, y, 1)), f"top{k}": R(topk_accuracy(cal, y, k)),
                        "ece_raw": R(expected_calibration_error(raw, y)), "ece_calibrated": R(expected_calibration_error(cal, y)),
                        "nll_calibrated": R(nll(cal, y))}

    # --- полный контур с памятью и без ---
    reps = pipe.run_batch(W, E)
    reps_nomem = pipe.run_batch(W, [None] * len(W))
    for name, rr in (("with_memory", reps), ("window_only", reps_nomem)):
        tc = tn = wc = wn = top1 = 0
        for rep, t in zip(rr, truth):
            best = max(rep.verdicts, key=lambda v: (v.support_ratio, v.hypothesis.confidence))
            top1 += best.hypothesis.scenario == t
            for v in rep.verdicts:
                ok = v.status is VerdictStatus.CONFIRMED
                if v.hypothesis.scenario == t:
                    tn += 1; tc += ok  # noqa: E702
                else:
                    wn += 1; wc += ok  # noqa: E702
        res.setdefault("verifier", {})[name] = {"true_confirmed": R(tc / tn), "wrong_confirmed": R(wc / wn),
                                                 "top1_after_verification": R(top1 / len(rr))}

    def outcome_block(rr):
        oc = Counter(r.outcome.value for r in rr)
        ident = [(r, t) for r, t in zip(rr, truth) if r.outcome is DiagnosisOutcome.IDENTIFIED]
        correct = sum(1 for r, t in ident if r.selected.scenario == t)
        per = defaultdict(Counter)
        for r, t in zip(rr, truth):
            per[t]["n"] += 1
            per[t][r.outcome.value] += 1
            if r.outcome is DiagnosisOutcome.IDENTIFIED:
                per[t]["identified_correct"] += r.selected.scenario == t
        return {
            "outcomes": share(oc, len(rr)),
            "identified_precision": R(correct / max(len(ident), 1)),
            "identified_coverage": R(len(ident) / len(rr)),
            "identified_wrong_share_of_all": R((len(ident) - correct) / len(rr)),
            "per_class": {c: {"n": v["n"], "identified": R(v["identified"] / v["n"]),
                              "identified_precision": R(v["identified_correct"] / max(v["identified"], 1)),
                              "ambiguous": R(v["ambiguous"] / v["n"]), "unconfirmed": R(v["unconfirmed"] / v["n"]),
                              "novel": R(v["novel"] / v["n"]), "no_anomaly": R(v["no_anomaly"] / v["n"])}
                          for c, v in sorted(per.items())},
        }

    res["outcomes"] = {"with_memory": outcome_block(reps), "window_only": outcome_block(reps_nomem)}
    res["novelty_false_alarm_on_test"] = R(np.mean([r.novelty.novel for r in reps]))

    # --- пробы ---
    no_event = set(cfg.raw["scenarios"]["no_event"])  # остальные пробы — вне области (ожидается NOVEL)
    pW, pE, pR = collect(ds, ds.split("probe"))
    preps = pipe.run_batch(pW, pE)
    pb = defaultdict(Counter)
    for r, run in zip(preps, pR):
        s = labels.scenario(run)
        pb[s]["n"] += 1
        pb[s][r.outcome.value] += 1
    probes = {}
    for s, c in sorted(pb.items()):
        expected = "no_anomaly" if s in no_event else "novel"
        probes[s] = {"group": "no_event" if s in no_event else "out_of_scope", "windows": c["n"],
                     "expected": expected, "as_expected": R(c[expected] / c["n"]), "identified": R(c["identified"] / c["n"]),
                     "outcomes": share(Counter({k: v for k, v in c.items() if k != "n"}), c["n"])}
    res["probes"] = probes

    # --- стресс-тест изменения мощности ---
    # основной режим: начало события обнаруживается причинно (D-039);
    # оракульный (момент ввода неисправности из манифеста) — только для сравнения
    vp_runs = load_vp_runs()

    def stress(gate):
        vp = {}
        for run in vp_runs:
            ws = vp_windows(run, ds)
            st = stationary_points(run, gate) if gate is not None else None
            rr = pipe.run_batch(ws, [vp_episode(run, ds, w.t_end, gate=gate, stationary=st) for w in ws])
            inj = run.injection_time_s
            pre = [r for w, r in zip(ws, rr) if inj is None or w.t_end < inj]
            post = [r for w, r in zip(ws, rr) if inj is not None and w.t_end >= inj]
            vp[run.scenario] = {
                "fault": run.fault_code, "windows_before": len(pre), "windows_after": len(post),
                "false_identified_before": R(np.mean([r.outcome is DiagnosisOutcome.IDENTIFIED for r in pre])) if pre else None,
                "before_outcomes": share(Counter(r.outcome.value for r in pre), len(pre)) if pre else None,
                "after_outcomes": share(Counter(r.outcome.value for r in post), len(post)) if post else None,
                "after_correct_identified": R(np.mean([r.outcome is DiagnosisOutcome.IDENTIFIED and r.selected.scenario == run.fault_code for r in post])) if post else None,
                "after_wrong_identified": R(np.mean([r.outcome is DiagnosisOutcome.IDENTIFIED and r.selected.scenario != run.fault_code for r in post])) if post else None,
            }
        pre_n = sum(v["windows_before"] for v in vp.values())
        post_n = sum(v["windows_after"] for v in vp.values())
        vp["summary"] = {
            "false_identified_before": R(sum(v["false_identified_before"] * v["windows_before"] for v in vp.values()) / pre_n),
            "wrong_identified_after": R(sum((v["after_wrong_identified"] or 0) * v["windows_after"] for v in vp.values()) / post_n),
            "correct_identified_after": R(sum((v["after_correct_identified"] or 0) * v["windows_after"] for v in vp.values()) / post_n),
            "windows_before": pre_n, "windows_after": post_n,
        }
        return vp

    res["variable_power"] = stress(pipe.gate)
    res["variable_power_oracle_onset"] = stress(None)
    # согласованность: на test NPPAD детектор начала должен давать t ≈ 0 (неисправность в 0,5 с)
    agree = 0
    for run_id in ds.split("test"):
        t_, raw_ = ds.run_raw(run_id)
        st = stationary_points(VPRun("nppad", None, None, t_, raw_), pipe.gate)
        ns = np.flatnonzero(~st)
        agree += int(len(ns) > 0 and ns[0] <= 1)
    res["onset_detector_agreement_test"] = R(agree / len(ds.split("test")))

    # --- верификатор ответа на детерминированных ответах + контрольные испорченные ответы ---
    analyst = AnalystAgent(llm_client=object())  # LLM не вызывается (use_llm=False)
    decisions = Counter()
    per_outcome = defaultdict(Counter)
    tamper = Counter()
    released_bad = 0
    replays = {}
    sample = list(range(0, len(W), answer_sample_every))
    for i in sample:
        w, rep, run = W[i], reps[i], runs[i]
        src = replays.setdefault(run, TelemetryReplay(ds, run))
        resolver = EvidenceResolver().register(EvidenceKind.TELEMETRY, src.resolves)
        if pipe.verifier.analogues is not None:
            resolver.register(EvidenceKind.HISTORICAL_EVENT, pipe.verifier.analogues.has_event)
        av = AnswerVerifier(resolver=resolver)
        if pipe.verifier.analogues is not None:  # аналоги для ответа (в run_batch не подбирались)
            rep = pipe.run(w, E[i])
        resp = analyst.explain_diagnosis(rep, cfg.scenario_titles, (w.t_start, w.t_end))
        d = av.verify("Что происходит на станции?", resp).decision.effect
        decisions[d.value] += 1
        per_outcome[rep.outcome.value][d.value] += 1
        claims = resp.metadata["verification"]["claims"]
        for kind in ("no_refs", "dangling_ref", "directive"):
            bad = copy.deepcopy(resp)
            cl = copy.deepcopy(claims)
            if kind == "no_refs":
                for c in cl:
                    c["evidence"] = []
            elif kind == "dangling_ref":
                cl[0]["evidence"] = [{"ref_id": "deadbeefdeadbeef#P", "kind": "telemetry"}]
            else:
                cl.append({"claim_id": "x", "text": "Необходимо снизить мощность реактора.",
                           "evidence": cl[0]["evidence"]})
            bad.metadata = {"verification": {"claims": cl}}
            eff = av.verify("q", bad).decision.effect
            tamper[f"{kind}:{eff.value}"] += 1
            released_bad += eff is PolicyEffect.ALLOW
    res["answer_verifier"] = {
        "pack": AnswerVerifier().pack.pack_id, "sampled_answers": len(sample),
        "decisions": share(decisions, len(sample)),
        "decisions_by_outcome": {o: dict(c) for o, c in sorted(per_outcome.items())},
        "tampered_answers": dict(sorted(tamper.items())), "tampered_released_as_allow": released_bad,
    }

    # --- аналоги ---
    f = pipe.verifier.analogues
    if f is not None:
        tot = Counter()
        for scen, spec in f.mapping["scenario"].items():
            target = {i.lower() for i in spec.get("initiating_events", [])}
            if not target or not spec.get("query"):
                continue
            for mode, use in (("text", False), ("combined", True)):
                got = f.find(scen, k=3, use_initiating_events=use)
                lab = [a for a in got if next(e for e in f.events if e["event_id"] == a.event_id).get("initiating_event")]
                rel = [a for a in lab if next(e for e in f.events if e["event_id"] == a.event_id)["initiating_event"].lower() in target]
                tot[f"{mode}_labelled"] += len(lab)
                tot[f"{mode}_relevant"] += len(rel)
        res["analogues"] = {m: {"precision_at_3_labelled": R(tot[f"{m}_relevant"] / max(tot[f"{m}_labelled"], 1)),
                                "labelled": tot[f"{m}_labelled"]} for m in ("text", "combined")}
    return res


def latency(ds, pipe, n: int = 300) -> dict:
    runs = ds.split("test")
    lat = defaultdict(list)
    count = 0
    for r in runs[:: max(1, len(runs) // 30)]:
        for w in list(ds.windows(r))[::5]:
            t0 = time.perf_counter()
            rep = pipe.run(w, ds.episode(r, w.t_end))
            total = (time.perf_counter() - t0) * 1000
            for k_, v in rep.latency_ms.items():
                lat[k_].append(v)
            lat["total_with_analogues"].append(total)
            count += 1
            if count >= n:
                break
        if count >= n:
            break
    return {k_: {"p50_ms": R(np.percentile(v, 50)), "p95_ms": R(np.percentile(v, 95))} for k_, v in lat.items()} | {"windows": count}


def versions(cfg, ds, pipe) -> dict:
    lock = ROOT / "uv.lock"
    return {
        **pipe.versions,
        "rulepack": AnswerVerifier().pack.pack_id,
        "data_source_commit": ds.manifest["source"]["commit"],
        "data_config_fingerprint": ds.manifest["config_fingerprint"],
        "config_version": cfg.version,
        "config_fingerprint_full": cfg.fingerprint(),
        "uv_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest()[:16],
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "seed": cfg.seed,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--repeat", action="store_true", help="повторить оценку и сравнить результаты")
    ap.add_argument("--answer-sample-every", type=int, default=25)
    args = ap.parse_args()
    cfg = default_config()
    ds = NPPADDataset()
    labels = LabelStore.load(ds.directory)
    pipe = DiagnosticPipeline.load()

    t0 = time.time()
    result = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "split": "test"}
    result["versions"] = versions(cfg, ds, pipe)
    result["metrics"] = evaluate(cfg, ds, labels, pipe, args.answer_sample_every)
    result["latency"] = latency(ds, pipe)
    result["runtime_s"] = round(time.time() - t0, 1)
    if args.repeat:
        again = evaluate(cfg, ds, labels, DiagnosticPipeline.load(), args.answer_sample_every)
        result["reproducibility"] = {"identical_metrics_on_repeat": again == result["metrics"]}
    text = json.dumps(result, ensure_ascii=False, indent=1)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
