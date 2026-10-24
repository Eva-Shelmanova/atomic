#!/usr/bin/env python3
"""
Оценка качества подбора аналогов по размеченному полю «Initiating Event» ETHZ (этап B2/C).

    uv run python scripts/eval_analogues.py [--out docs/test-results/<файл>.json]

Текстовый поиск (BM25 без структурного поля) оценивается по разметке: аналог релевантен,
если его тип исходного события входит в список, заданный для сценария в config/analogues.toml.
precision@k считается только по событиям с размеченным типом; доля неразмеченных указывается отдельно.
recall@k — доля найденных среди всех размеченных релевантных событий базы (как в 6-м семестре).
Совмещённый режим использует то же поле Initiating Event, что и разметка, поэтому его оценка
частично замкнута на себя; независимая мера качества поиска — текстовый режим.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atomic.diagnostics.analogues import AnalogueFinder  # noqa: E402


def at_k(f, ie_of: dict, ks: list[int]) -> dict:
    """Средние по сценариям precision@k (по размеченным) и recall@k для обоих режимов."""
    out = {}
    for mode, use_ie in (("text", False), ("combined", True)):
        prec, rec = {k: [] for k in ks}, {k: [] for k in ks}
        for scen, spec in f.mapping["scenario"].items():
            target = {i.lower() for i in spec.get("initiating_events", [])}
            if not target or not spec.get("query"):
                continue
            relevant_all = sum(1 for v in ie_of.values() if v and v.lower() in target)
            res = f.find(scen, k=max(ks), use_initiating_events=use_ie)
            for k in ks:
                top = [a for a in res[:k] if ie_of[a.event_id]]
                hit = sum(1 for a in top if ie_of[a.event_id].lower() in target)
                if top:
                    prec[k].append(hit / len(top))
                if relevant_all:
                    rec[k].append(hit / relevant_all)
        out[mode] = {f"precision@{k}": round(sum(prec[k]) / len(prec[k]), 3) if prec[k] else None for k in ks}
        out[mode].update({f"recall@{k}": round(sum(rec[k]) / len(rec[k]), 3) if rec[k] else None for k in ks})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--ks", default="3,10,20", help="k для precision@k / recall@k")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    f = AnalogueFinder.load()
    ie_of = {e["event_id"]: e.get("initiating_event") for e in f.events}
    rows, tot = {}, {"text": [0, 0, 0], "combined": [0, 0, 0]}
    for scen, spec in f.mapping["scenario"].items():
        target = {i.lower() for i in spec.get("initiating_events", [])}
        if not target or not spec.get("query"):
            continue
        row = {"available": sum(1 for v in ie_of.values() if v and v.lower() in target)}
        for mode, use_ie in (("text", False), ("combined", True)):
            res = f.find(scen, k=args.k, use_initiating_events=use_ie)
            labelled = [a for a in res if ie_of[a.event_id]]
            rel = [a for a in labelled if ie_of[a.event_id].lower() in target]
            row[mode] = {"returned": len(res), "labelled": len(labelled), "relevant": len(rel),
                         f"precision@{args.k}_labelled": round(len(rel) / len(labelled), 3) if labelled else None}
            tot[mode][0] += len(res); tot[mode][1] += len(labelled); tot[mode][2] += len(rel)
        rows[scen] = row
        print(f"{scen:6s} доступно {row['available']:3d} | текст: {row['text']['relevant']}/{row['text']['labelled']} размеч. из {row['text']['returned']} | "
              f"совмещ.: {row['combined']['relevant']}/{row['combined']['labelled']} из {row['combined']['returned']}")
    summary = {m: {"returned": t[0], "labelled": t[1], "relevant": t[2],
                   "precision_labelled": round(t[2] / t[1], 3) if t[1] else None} for m, t in tot.items()}
    summary["at_k"] = at_k(f, ie_of, [int(x) for x in args.ks.split(",")])
    print("Итого:", summary)
    if args.out:
        Path(args.out).write_text(json.dumps({"k": args.k, "per_scenario": rows, "summary": summary, "version": f.version},
                                             ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
