#!/usr/bin/env python3
"""
Оценка пояснений AnalystAgent на реальной LLM (Ollama) — этап C.

    uv run python scripts/eval_llm_narrative.py [--per-class 1] [--out docs/test-results/<файл>.json]

По одному (или --per-class) test-окну каждого класса + пробы: diagnose(use_llm=True).
Измеряется: доля предложений со ссылками [E…]/[A…], несуществующие ссылки, указания к действию,
решение верификатора ответа, время ответа LLM. Требует запущенную Ollama.
"""

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atomic.config import MODEL_ANALYST  # noqa: E402
from atomic.diagnostics.config import default_config  # noqa: E402
from atomic.diagnostics.labels import LabelStore  # noqa: E402
from atomic.diagnostics.nppad import NPPADDataset  # noqa: E402
from atomic.diagnostics.telemetry import TelemetryReplay  # noqa: E402
from atomic.memory.vector import VectorMemory  # noqa: E402
from atomic.orchestration.workflow import AtomicWorkflow  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = default_config()
    ds = NPPADDataset()
    labels = LabelStore.load(ds.directory)
    wf = AtomicWorkflow(vector_memory=VectorMemory(), verification_mode="shadow")
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
        rows.append({"scenario": s, "outcome": wf.last_diagnosis.outcome.value, "decision": d.effect.value,
                     "violated": list(d.violated_rule_ids), "narrative": st, "seconds": round(sec, 1),
                     "narrative_rejected": None if rej is None else rej["decision"]["violated_rule_ids"],
                     "narrative_text": (r.response.metadata.get("narrative_text") or "")[:1200]})
        tot.update({k: v for k, v in st.items()})
        dec[d.effect.value] += 1
        rejected += rej is not None
        print(f"{s:6s} исход {rows[-1]['outcome']:11s} решение {d.effect.value:13s} нарушено {list(d.violated_rule_ids)} "
              f"предложений {st.get('sentences')} со ссылками {st.get('cited')} пояснение отклонено {rows[-1]['narrative_rejected']} ({sec:.0f} с)")
    summary = {"cases": len(rows), "decisions": dict(dec), "sentences": tot["sentences"], "cited": tot["cited"],
               "cited_share": round(tot["cited"] / max(tot["sentences"], 1), 3), "unknown_citations": tot["unknown_citations"],
               "narratives_rejected": rejected,
               "model": MODEL_ANALYST}
    print("Итого:", summary)
    if args.out:
        Path(args.out).write_text(json.dumps({"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                              "summary": summary, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
