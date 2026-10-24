#!/usr/bin/env python3
"""
Демонстрация левого контура для одного окна NPPAD (иллюстрация для отчёта).

    uv run python scripts/demo_diagnosis.py --scenario SGBTR --window 5 [--split val]

Окно → барьер (6 каналов) → генератор (top-k) → верификатор (все наблюдаемые каналы)
→ аналоги ETHZ. Истинный сценарий берётся из меток только для подписи в выводе:
в рабочий путь он не передаётся.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atomic.diagnostics.analogues import AnalogueFinder  # noqa: E402
from atomic.diagnostics.barrier import InformationBarrier  # noqa: E402
from atomic.diagnostics.config import default_config  # noqa: E402
from atomic.diagnostics.generator import HypothesisGenerator  # noqa: E402
from atomic.diagnostics.hypothesis_verifier import HypothesisVerifier  # noqa: E402
from atomic.diagnostics.labels import LabelStore  # noqa: E402
from atomic.diagnostics.nppad import NPPADDataset  # noqa: E402
from atomic.diagnostics.signatures import ScenarioSignatures  # noqa: E402

STATUS = {"confirmed": "ПОДТВЕРЖДЕНА", "rejected": "ОТКЛОНЕНА", "uncertain": "НЕ ОПРЕДЕЛЕНО"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="SGBTR")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--split", default="val", choices=["train", "val", "probe"])
    ap.add_argument("--run", type=int, default=0, help="номер прогона сценария в выборке")
    args = ap.parse_args()

    cfg = default_config()
    ds = NPPADDataset()
    labels = LabelStore.load(ds.directory)
    runs = [r for r in ds.split(args.split) if labels.scenario(r) == args.scenario]
    run = runs[args.run]
    w = list(ds.windows(run))[args.window]
    gen = HypothesisGenerator.load(cfg.resolve("models") / "generator")
    ver = HypothesisVerifier(ScenarioSignatures.load(cfg.resolve("models") / "verifier"), cfg.verifier, AnalogueFinder.load())
    rw = InformationBarrier().restrict(w)
    hs = gen.generate(rw)
    titles = cfg.scenario_titles

    lab = labels.get(run)
    print(f"Окно {w.window_id}: {w.t_start:.0f}–{w.t_end:.0f} с от начала прогона "
          f"[для подписи: истинный сценарий {lab.scenario}, тяжесть {lab.severity:g} %]")
    print(f"Генератор видит каналы: {', '.join(rw.channels)}")
    print(f"Генератор {hs.generator_version}:")
    for h in hs.hypotheses:
        print(f"  {h.rank}. {h.describe(titles)}")
    print(f"\nВерификатор {ver.version.split('|')[0]} (каналов в сигнатурах: {len(ver.sig.channels)}):")
    for v in ver.verify(w, hs):
        h = v.hypothesis
        print(f"\n■ {h.scenario} — {STATUS[v.status.value]}, доля подтверждений {v.support_ratio:.0%} "
              f"(за: {len(v.evidence_for)}, против: {len(v.evidence_against)})")
        for note in v.notes:
            print(f"   ! {note}")
        for e in sorted(v.evidence_for, key=lambda e: -e.weight)[:3]:
            print(f"   + {e.detail} [вес {e.weight:.2f}, ссылка {e.ref_id}]")
        for e in sorted(v.evidence_against, key=lambda e: -e.weight)[:3]:
            print(f"   − {e.detail} [вес {e.weight:.2f}, ссылка {e.ref_id}]")
        for a in v.analogues[:2]:
            print(f"   ≈ {a.describe()[:180]} [{a.event_id}]")
    print("\nСистема не формирует рекомендаций действий: решение остаётся за оператором.")


if __name__ == "__main__":
    main()
