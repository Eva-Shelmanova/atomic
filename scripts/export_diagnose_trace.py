#!/usr/bin/env python3
"""
Живая трасса диагностического контура в Langfuse (требование 10, этап E).

    docker compose -f langfuse/docker-compose.yml up -d
    uv run python scripts/export_diagnose_trace.py [--scenario LOCA] [--t 600] [--out docs/test-results/<файл>.json]

Выполняет AtomicWorkflow.diagnose на val-окне с включённой трассировкой, читает трассу обратно через
публичный API Langfuse и проверяет, что по ней восстанавливается путь
«данные → гипотеза → проверка → ответ → решение». Ключи берутся из .env и в выгрузку не попадают.
"""

import argparse
import base64
import json
import os
import sys
import time
import urllib.request
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from atomic.diagnostics.labels import LabelStore  # noqa: E402
from atomic.diagnostics.nppad import NPPADDataset  # noqa: E402
from atomic.diagnostics.telemetry import TelemetryReplay  # noqa: E402
from atomic.memory.scenario_kb import seed_scenario_kb  # noqa: E402
from atomic.memory.vector import VectorMemory  # noqa: E402
from atomic.orchestration.workflow import AtomicWorkflow  # noqa: E402

PATH_SPANS = ("diagnostics.hypotheses", "diagnostics.verification", "diagnostics.novelty", "diagnostics.documents")


def api(path: str) -> dict:
    base = os.environ.get("LANGFUSE_BASE_URL") or os.environ.get("LANGFUSE_HOST") or "http://localhost:3000"
    token = base64.b64encode(f"{os.environ['LANGFUSE_PUBLIC_KEY']}:{os.environ['LANGFUSE_SECRET_KEY']}".encode()).decode()
    req = urllib.request.Request(f"{base.rstrip('/')}{path}", headers={"Authorization": f"Basic {token}"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.load(r)


def _parsed(v):
    """Вложенные метаданные OpenTelemetry могут прийти строкой JSON."""
    if isinstance(v, str) and v[:1] in "{[":
        try:
            return json.loads(v)
        except ValueError:
            return v
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="LOCA")
    ap.add_argument("--t", type=float, default=600.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not os.environ.get("LANGFUSE_SECRET_KEY"):
        sys.exit("LANGFUSE_SECRET_KEY не задан (.env)")
    ds = NPPADDataset()
    labels = LabelStore.load(ds.directory)
    run = next(r for r in ds.split("val") if labels.scenario(r) == args.scenario)
    wf = AtomicWorkflow(vector_memory=VectorMemory(), verification_mode="shadow")
    seed_scenario_kb(wf.vector_memory)
    session = f"stage-e-trace-{uuid.uuid4().hex[:8]}"
    r = wf.diagnose(TelemetryReplay(ds, run), args.t, session_id=session)

    trace = None
    for _ in range(30):  # приём в Langfuse асинхронный
        time.sleep(2)
        found = api(f"/api/public/traces?sessionId={session}&limit=1").get("data", [])
        if found:
            trace = api(f"/api/public/traces/{found[0]['id']}")
            # спаны этапов приходят раньше итоговых метаданных корневого спана
            if len(trace.get("observations", [])) > len(PATH_SPANS) and "verification" in (trace.get("metadata") or {}):
                break
    if trace is None:
        sys.exit("трасса не появилась в Langfuse")
    spans = {o["name"]: {**o, "output": _parsed(o.get("output"))} for o in trace.get("observations", [])}
    meta = {k: _parsed(v) for k, v in (trace.get("metadata") or {}).items()}
    bundle = meta.get("verification") or {}
    path = {
        "data (window_id)": meta.get("window_id"),
        "hypotheses": [h["scenario"] for h in (spans.get("diagnostics.hypotheses", {}).get("output") or {}).get("hypotheses", [])],
        "verification": [(v["hypothesis"]["scenario"], v["status"]) for v in spans.get("diagnostics.verification", {}).get("output") or []],
        "novelty": (spans.get("diagnostics.novelty", {}).get("output") or {}).get("novel"),
        "documents": spans.get("diagnostics.documents", {}).get("output"),
        "outcome": meta.get("diagnosis_outcome"),
        "claims": len(bundle.get("claims", [])),
        "decision": (bundle.get("decision") or {}).get("effect"),
        "rulepack": (bundle.get("decision") or {}).get("applied_pack_id"),
    }
    ok = all(k in spans for k in PATH_SPANS) and path["decision"] == r.bundle.decision.effect.value and bool(path["data (window_id)"])
    for k, v in path.items():
        print(f"{k:18s} {v}")
    print("путь восстанавливается по трассе:", ok)
    if args.out:
        Path(args.out).write_text(json.dumps({
            "trace_id": trace["id"], "name": trace.get("name"), "session_id": session, "timestamp": trace.get("timestamp"),
            "langfuse_version": api("/api/public/health").get("version"), "observations": sorted(spans),
            "reconstructed_path": path, "path_complete": ok, "metadata_keys": sorted(meta),
        }, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
