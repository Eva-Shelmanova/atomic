#!/usr/bin/env python3
"""
Подготовка базы исторических событий ETHZ для поиска аналогов (этап B2).

    uv run python scripts/load_ethz.py

Источник: ETHDB/ETHZ Curated Nuclear Events Database/Support_file.xlsx,
лист «ReMerged DB_public (MOST CURRENT)» — 1561 событие. Это тот же лист, из которого
VBA-макрос ETHZ_Database_Access_tool.xlsm заполняет свои листы Events и Failure Sequences
(сам инструмент распространяется с пустыми листами, D-027).
Фильтр (D-020): Industry = Energy, Facility = Plant, OMIT ≠ 1, Type = PWR.
Кроме описания берутся структурированные поля, которые использует инструмент:
тип исходного события, система, компонент, частичный отказ и цепочка отказов.
Результат: data/processed/ethz/events.json.
"""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atomic.diagnostics.config import default_config  # noqa: E402

SOURCE = ROOT / "ETHDB" / "ETHZ Curated Nuclear Events Database" / "Support_file.xlsx"
SHEET_PREFIX = "ReMerged DB_public"
FIELDS = {
    "EventNumber": "number",
    "Date": "date",
    "Country": "country",
    "Site": "site",
    "Unit": "unit",
    "Type": "reactor_type",
    "Mode": "mode",
    "INES (assessed)": "ines",
    "Description": "description",
    "Origin_description": "origin",
    "Cause_description": "cause",
    "Core relevant": "core_relevant",
    "Initiating Event": "initiating_event",
    "Involved Initiating Event": "involved_initiating_event",
    "System": "system",
    "Component": "component",
    "Short Description": "short_description",
    "Partial Failure": "partial_failure",
}
# Цепочка отказов (как в листе Failure Sequences инструмента): уровень → система.
SEQUENCE_LEVELS = {"0": "0", "11": "11", "21": "21", "31": "31", "12": "12", "22": "22"}


def canonical_ie(value):
    """Единое написание типа исходного события («Steam Generator Tube rupture» → «… Rupture»)."""
    if value is None:
        return None
    v = " ".join(str(value).split())
    if not v or v.lower() == "no":
        return None
    known = {
        "steam generator tube rupture": "Steam Generator Tube Rupture",
        "loss of main feedwater": "Loss of Main Feedwater",
        "internal flooding": "Flooding Internal",
    }
    return known.get(v.lower(), v)


def clean(v):
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.date().isoformat()
    s = str(v).strip()
    return s or None


def main() -> None:
    wb = openpyxl.load_workbook(SOURCE, read_only=True, data_only=True)
    ws = next(w for w in wb.worksheets if w.title.startswith(SHEET_PREFIX))
    rows = ws.iter_rows(values_only=True, max_col=170)
    header = next(rows)
    col = {h: i for i, h in enumerate(header) if h}
    total, kept, reasons = 0, [], {}
    for r in rows:
        if not any(r):
            continue
        total += 1
        why = None
        if clean(r[col["Industry"]]) != "Energy":
            why = "industry"
        elif clean(r[col["Facility"]]) != "Plant":
            why = "facility"
        elif str(r[col["OMIT"]]).strip().startswith("1"):
            why = "omit"
        elif clean(r[col["Type"]]) != "PWR":
            why = "reactor_type"
        elif not clean(r[col["Description"]]):
            why = "no_description"
        if why:
            reasons[why] = reasons.get(why, 0) + 1
            continue
        ev = {dst: clean(r[col[src]]) for src, dst in FIELDS.items()}
        ev["initiating_event"] = canonical_ie(ev["initiating_event"])
        ev["involved_initiating_event"] = canonical_ie(ev["involved_initiating_event"])
        ev["failure_sequence"] = {lvl: clean(r[col[c]]) for lvl, c in SEQUENCE_LEVELS.items() if c in col and clean(r[col[c]])}
        ev["event_id"] = f"ethz:{ev['number']}"
        kept.append(ev)
    kept.sort(key=lambda e: int(float(e["number"])))
    out = default_config().resolve("ethz_events")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": "ETH Zurich Curated Nuclear Events Database (release 2020-11-17), Support_file.xlsx",
        "sheet": ws.title,
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest()[:16],
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "filter": "Industry=Energy, Facility=Plant, OMIT!=1, Type=PWR",
        "total_rows": total,
        "excluded": reasons,
        "events": kept,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    from collections import Counter

    ies = Counter(e["initiating_event"] for e in kept if e["initiating_event"])
    payload["initiating_events"] = dict(ies.most_common())
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Всего строк: {total}; оставлено PWR-событий: {len(kept)}; исключено: {reasons}")
    print(f"С размеченным исходным событием: {sum(ies.values())}: {dict(ies.most_common())}")
    print(f"Сохранено: {out}")


if __name__ == "__main__":
    main()
