#!/usr/bin/env python3
"""
Конвертация Variable_Power_Data (только .mdb) в CSV формата Operation_csv_data (D-029).

    uv run python scripts/convert_variable_power.py

Читает PlotData из case1.mdb каждого из 12 сценариев чистым Python (access-parser,
без mdbtools/ODBC), сортирует по времени (в исходных базах порядок нарушен),
пишет NPPAD/raw/Variable_Power_csv/<сценарий>/case1.csv и манифест с временем
ввода неисправности из Variable_Power_Data/manifest.csv.
"""

import csv
import json
import sys
from pathlib import Path

from access_parser import AccessParser

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "NPPAD" / "raw" / "Variable_Power_Data"
DST = ROOT / "NPPAD" / "raw" / "Variable_Power_csv"


def main() -> None:
    meta = {r["scenario_id"]: r for r in csv.DictReader((SRC / "manifest.csv").open(encoding="utf-8"))}
    out_manifest = {}
    for scen in sorted(meta):
        table = AccessParser(str(SRC / scen / "case1.mdb")).parse_table("PlotData")
        cols = [c for c in table if c]
        n = len(table["TIME"])
        rows = sorted(zip(*[table[c] for c in cols]), key=lambda r: r[cols.index("TIME")])
        (DST / scen).mkdir(parents=True, exist_ok=True)
        with (DST / scen / "case1.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(cols)
            w.writerows(rows)
        times = [r[cols.index("TIME")] for r in rows]
        m = meta[scen]
        inj = None if m["injection_time_s"] in ("NA", "") else float(m["injection_time_s"])
        out_manifest[scen] = {
            "category": m["category"], "fault_code": None if m["fault_code"] == "NA" else m["fault_code"],
            "injection_time_s": inj, "start_power_pct": float(m["start_power_pct"]), "end_power_pct": float(m["end_power_pct"]),
            "rows": n, "t_min": times[0], "t_max": times[-1],
            "unique_steps": sorted({round(b - a, 1) for a, b in zip(times, times[1:])})[:6],
        }
        print(f"{scen:16s} строк {n:4d}, t {times[0]:.0f}–{times[-1]:.0f} с, шаги {out_manifest[scen]['unique_steps']}, ввод {inj}")
    (DST / "manifest.json").write_text(json.dumps(out_manifest, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
