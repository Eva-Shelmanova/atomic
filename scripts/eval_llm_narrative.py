#!/usr/bin/env python3
"""
Оценка пояснений AnalystAgent на реальной LLM (Ollama) — этап C.

    uv run python scripts/eval_llm_narrative.py [--per-class 1] [--out docs/test-results/<файл>.json]

По одному (или --per-class) test-окну каждого класса + пробы: diagnose(use_llm=True).
Измеряется: доля предложений со ссылками [E…]/[A…], несуществующие ссылки, указания к действию,
решение верификатора ответа, время ответа LLM. Требует запущенную Ollama.

--judge MODEL — LLM-as-a-Judge (deepeval, как в 6-м семестре): FaithfulnessMetric (пояснение не
противоречит детерминированному отчёту) и AnswerRelevancyMetric. Судья — модель другого семейства,
чем аналитик. Оценка вспомогательная: на выпуск ответа не влияет. Две контрольные пары (верное и
искажённое пояснение) показывают, различает ли судья качество.
"""

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atomic.config import MODEL_ANALYST, ollama_model_versions  # noqa: E402
from atomic.diagnostics.config import default_config  # noqa: E402
from atomic.diagnostics.labels import LabelStore  # noqa: E402
from atomic.diagnostics.nppad import NPPADDataset  # noqa: E402
from atomic.diagnostics.telemetry import TelemetryReplay  # noqa: E402
from atomic.memory.scenario_kb import seed_scenario_kb  # noqa: E402
from atomic.memory.vector import VectorMemory  # noqa: E402
from atomic.orchestration.workflow import AtomicWorkflow  # noqa: E402


JUDGE_DEFAULT = "vikhr-llama3.1-8b-instruct-r-21-09-24-Q4_K_M:latest"
NARRATIVE_MARK = "\n\nПояснение (сформулировано"
JUDGE_QUERY = "Объясни оператору, что показывает автоматическая проверка гипотез о состоянии АЭС."
CONTROL_REPORT = (
    "Оценка ситуации: Вероятна ситуация LOCA (течь теплоносителя, горячая нитка): подтверждена независимой проверкой.\n"
    "[E1] Press Partial RB air (bar): уровень +2.11σ — в типичном диапазоне для LOCA\n"
    "[E2] Flow Containment spray (t/hr): уровень +1.67σ — в типичном диапазоне для LOCA\n"
    "[E3] Press Reactor building (bar): уровень +1.86σ — вне допустимого диапазона для SGBTR"
)
CONTROL_NARRATIVES = {
    "faithful": "Давление в защитной оболочке повышено, что типично для течи в горячей нитке [E1]. "
                "Расход спринклерной системы также соответствует LOCA [E2]. Повышенное давление в здании "
                "реактора не согласуется с разрывом трубки ПГ [E3].",
    "distorted": "Давление в защитной оболочке упало до нуля [E1]. Спринклерная система не работает [E2]. "
                 "Ситуация соответствует разрыву трубки ПГ B [E3].",
}


def mean_score(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


def make_judge(model: str):
    import os

    os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "300")
    os.environ.setdefault("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", "600")
    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
    from deepeval.models import OllamaModel
    from deepeval.test_case import LLMTestCase

    from atomic.config import OLLAMA_BASE_URL

    llm = OllamaModel(model=model, base_url=OLLAMA_BASE_URL)

    def judge(narrative: str, report_text: str) -> dict:
        case = LLMTestCase(input=JUDGE_QUERY, actual_output=narrative,
                           retrieval_context=[ln.strip() for ln in report_text.splitlines() if ln.strip()])
        out = {}
        for name, metric in (("faithfulness", FaithfulnessMetric(model=llm, include_reason=True, async_mode=False)),
                             ("relevancy", AnswerRelevancyMetric(model=llm, include_reason=True, async_mode=False))):
            try:
                metric.measure(case)
                out[name] = {"score": round(float(metric.score), 3), "reason": (metric.reason or "")[:400]}
            except Exception as e:  # судья вернул невалидный JSON и т.п. — оценка пропускается
                out[name] = {"score": None, "reason": f"ошибка судьи: {e!s:.200}"}
        return out

    return judge


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--judge", nargs="?", const=JUDGE_DEFAULT, default=None, help="модель-судья Ollama")
    args = ap.parse_args()
    judge = make_judge(args.judge) if args.judge else None
    cfg = default_config()
    ds = NPPADDataset()
    labels = LabelStore.load(ds.directory)
    wf = AtomicWorkflow(vector_memory=VectorMemory(), verification_mode="shadow")
    seed_scenario_kb(wf.vector_memory)
    cases = []
    for s in list(cfg.benchmark_scenarios) + ["TT", "MD"]:
        split = "probe" if s in ("TT", "MD") else "test"
        cases += [(s, r) for r in [x for x in ds.split(split) if labels.scenario(x) == s][: args.per_class]]
    rows, tot, dec, rejected = [], Counter(), Counter(), 0
    for s, run in cases:
        src = TelemetryReplay(ds, run)
        t = src.windows[min(8, len(src.windows) - 1)].t_end
        t0 = time.perf_counter()
        r = wf.diagnose(src, t, use_llm=True)
        sec = time.perf_counter() - t0
        st = r.response.metadata.get("narrative_stats") or {}
        d = r.bundle.decision
        rej = r.response.metadata.get("narrative_rejected")
        narrative = r.response.metadata.get("narrative_text") or ""
        report_text = r.response.content.split(NARRATIVE_MARK)[0]
        scores = judge(narrative, report_text) if judge and narrative else None
        rows.append({"scenario": s, "outcome": wf.last_diagnosis.outcome.value, "decision": d.effect.value,
                     "judge": scores,
                     "violated": list(d.violated_rule_ids), "narrative": st, "seconds": round(sec, 1),
                     "narrative_rejected": None if rej is None else rej["decision"]["violated_rule_ids"],
                     "narrative_text": (r.response.metadata.get("narrative_text") or "")[:1200]})
        tot.update({k: v for k, v in st.items()})
        dec[d.effect.value] += 1
        rejected += rej is not None
        print(f"{s:6s} исход {rows[-1]['outcome']:11s} решение {d.effect.value:13s} нарушено {list(d.violated_rule_ids)} "
              f"предложений {st.get('sentences')} со ссылками {st.get('cited')} пояснение отклонено {rows[-1]['narrative_rejected']} "
              f"судья {scores and {k: v['score'] for k, v in scores.items()}} ({sec:.0f} с)")
    summary = {"cases": len(rows), "decisions": dict(dec), "sentences": tot["sentences"], "cited": tot["cited"],
               "cited_share": round(tot["cited"] / max(tot["sentences"], 1), 3), "unknown_citations": tot["unknown_citations"],
               "narratives_rejected": rejected,
               "model": MODEL_ANALYST, "model_digest": ollama_model_versions([MODEL_ANALYST])[MODEL_ANALYST]}
    if judge:
        judged = [r["judge"] for r in rows if r["judge"]]
        summary["judge"] = {
            "model": args.judge, "model_digest": ollama_model_versions([args.judge])[args.judge], "judged": len(judged),
            **{f"{m}_mean": mean_score([j[m]["score"] for j in judged]) for m in ("faithfulness", "relevancy")},
            "control": {name: judge(text, CONTROL_REPORT) for name, text in CONTROL_NARRATIVES.items()},
        }
    print("Итого:", summary)
    if args.out:
        Path(args.out).write_text(json.dumps({"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                              "summary": summary, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
