#!/usr/bin/env python3
"""
Справка по сценариям для базы знаний ассистента (этап E, D-043).

    uv run python scripts/build_scenario_kb.py

Источник — журналы событий симулятора PCTRAN («<тяжесть>Transient Report.txt») обучающих
прогонов NPPAD. Для каждого из обучаемых сценариев считается, какие действия систем станции
происходят (доля прогонов) и когда (медиана и межквартильный размах времени от ввода неисправности).
Test- и val-прогоны не используются. Выход — dataset/scenarios/<код>.txt: текст описательный,
без указаний к действию (их отсекает правило NO_DIRECTIVES).
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atomic.diagnostics.config import PROJECT_ROOT, default_config  # noqa: E402
from atomic.diagnostics.nppad import NPPADDataset  # noqa: E402
from atomic.memory.scenario_kb import SCENARIO_KB_DIR  # noqa: E402

RAW = PROJECT_ROOT / "NPPAD" / "raw" / "NPPAD"
LINE = re.compile(r"^\s*0*(\d+(?:\.\d+)?) sec, (.+?)\s*$")
MIN_SHARE = 0.2  # событие попадает в справку, если встречается хотя бы в 20 % прогонов

# Нормализованное событие PCTRAN → описание (номер насоса/клапана объединяется: «#»)
EVENTS = {
    "Reactor Scram": "аварийная остановка реактора (срабатывание АЗ)",
    "Scram Lo RX Press": "сигнал АЗ по низкому давлению первого контура",
    "Scram High Flux": "сигнал АЗ по превышению нейтронного потока",
    "HPSI start low RX Press": "запуск высоконапорного впрыска САОЗ по низкому давлению первого контура",
    "HPSI start high RB Press": "запуск высоконапорного впрыска САОЗ по высокому давлению в защитной оболочке",
    "HPI Pump # Position Change: 100%": "включение насосов высоконапорного впрыска",
    "Ctmt Vent Valve # Position Change: 0%": "закрытие клапана вентиляции защитной оболочки",
    "Ctmt Spray Starts": "запуск спринклерной системы защитной оболочки",
    "RBS Pump # Position Change: 100%": "включение насосов спринклерной системы",
    "FW isolation on low": "сигнал отсечения питательной воды по низкой температуре теплоносителя",
    "FWIV # Position Change: 0%": "закрытие отсечных клапанов питательной воды",
    "PZR Safety Relief Valve # Position Change: 100%": "открытие предохранительного клапана компенсатора давления",
    "PZR Safety Relief Valve # Position Change: 0%": "закрытие предохранительного клапана компенсатора давления",
    "Sump Recirc RWST lvl": "переход на рециркуляцию из приямка по уровню в баке запаса воды",
    "RCDT ruptured": "разрыв мембраны бака сбора протечек теплоносителя",
    "Accumulator Valve # Position Change: 100%": "открытие клапанов гидроёмкостей САОЗ",
    "Core Uncovered": "оголение активной зоны",
    "Low SG Level": "низкий уровень в парогенераторе",
    "MDAFW Pump # Position Change: 100%": "включение насосов аварийной питательной воды",
    "Safety Relief Valve # Position Change: 100%": "открытие предохранительных клапанов паропровода",
    "Safety Relief Valve # Position Change: 0%": "закрытие предохранительных клапанов паропровода",
    "RHR Pump # Position Change: 100%": "включение насосов отвода остаточного тепла",
}
SEVERITY_UNIT = {"LOCA": "% от 100 см²", "LOCAC": "% от 100 см²", "SLBIC": "% от 100 см²", "SLBOC": "% от 100 см²",
                 "FLB": "% от 100 см²", "SGATR": "% от разрыва одной трубки", "SGBTR": "% от разрыва одной трубки",
                 "RW": "% извлечения", "RI": "% ввода", "LLB": "%", "LR": "%"}


def normalize(event: str) -> str:
    event = re.sub(r"#\s*\d+", "#", event)
    if "Position Change" in event:
        return event
    return re.sub(r"\s*-?\d+(?:\.\d+)?\s*(bar|C|M3|M|%|cm|kg/s|t/hr|ppm|sec)?\s*$", "", event).strip()


def read_report(path: Path) -> dict[str, float]:
    """Первое время каждого события (с от ввода неисправности, которая вводится в 0,5 с)."""
    first: dict[str, float] = {}
    for ln in path.read_text(errors="replace").splitlines():
        m = LINE.match(ln)
        if not m or m.group(2).startswith("Malfunction"):
            continue
        ev = normalize(m.group(2))
        first.setdefault(ev, max(float(m.group(1)) - 0.5, 0.0))
    return first


def fmt_t(x: float) -> str:
    return f"{x:.0f} с" if x < 600 else f"{x / 60:.0f} мин"


def main() -> None:
    cfg = default_config()
    ds = NPPADDataset()
    labels = json.loads((ds.directory / "labels.json").read_text(encoding="utf-8"))
    manifest = json.loads((ds.directory / "manifest.json").read_text(encoding="utf-8"))
    commit = str(manifest["source"]["commit"])[:7]
    by_scen: dict[str, list[tuple[float, dict[str, float]]]] = defaultdict(list)
    missing = 0
    for rid in ds.split("train"):
        lab = labels[rid]
        scen, num = lab["source_file"].split("/")
        path = RAW / scen / f"{num.removesuffix('.csv')}Transient Report.txt"
        if not path.exists():
            missing += 1
            continue
        by_scen[scen].append((float(lab.get("severity") or 0.0), read_report(path)))
    SCENARIO_KB_DIR.mkdir(parents=True, exist_ok=True)
    unknown: set[str] = set()
    for scen in cfg.benchmark_scenarios:
        runs = by_scen.get(scen, [])
        if not runs:
            continue
        title = cfg.scenario_titles.get(scen, scen)
        sev = [s for s, _ in runs]
        times: dict[str, list[float]] = defaultdict(list)
        for _, rep in runs:
            for ev, t in rep.items():
                times[ev].append(t)
        rows = []
        for ev, ts in times.items():
            share = len(ts) / len(runs)
            if share < MIN_SHARE:
                continue
            if ev not in EVENTS:
                unknown.add(ev)
            q1, med, q3 = np.percentile(ts, [25, 50, 75])
            rows.append((med, f"- {EVENTS.get(ev, ev)} — {share * 100:.0f} % прогонов; {fmt_t(med)} ({fmt_t(q1)}–{fmt_t(q3)})"))
        rows.sort()
        scram = len(times.get("Reactor Scram", [])) / len(runs)
        lines = [
            f"Справка по сценарию {scen} — {title}.",
            "Источник: журналы событий симулятора PCTRAN (файлы «Transient Report») обучающих прогонов "
            f"набора NPPAD (thu-inet, коммит {commit}; число прогонов — {len(runs)}, тяжесть {min(sev):g}–{max(sev):g} "
            f"{SEVERITY_UNIT.get(scen, '%')}). Справка построена автоматически скриптом scripts/build_scenario_kb.py; "
            "val- и test-прогоны не использовались.",
            "",
        ]
        if rows:
            lines += [
                "Типичная последовательность действий систем станции после ввода неисправности "
                "(доля прогонов; медиана и межквартильный размах времени от ввода):",
                *[r for _, r in rows],
                "",
                f"Аварийная остановка реактора происходит в {scram * 100:.0f} % прогонов сценария {scen}.",
            ]
        else:
            lines.append(f"Ни в одном журнале обучающих прогонов сценария {scen} действия систем защиты "
                         "(АЗ, САОЗ, спринклерная система, отсечение питательной воды) не зарегистрированы.")
        (SCENARIO_KB_DIR / f"{scen}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"{scen:6s} прогонов {len(runs):3d}, событий в справке {len(rows)}")
    if unknown:
        print("Без перевода:", sorted(unknown))
    if missing:
        print(f"Журналов не найдено: {missing}")


if __name__ == "__main__":
    main()
