#!/usr/bin/env python3
"""
Демонстрация полного пути для одного момента прогона NPPAD (иллюстрация для отчёта).

    uv run python scripts/demo_diagnosis.py --scenario LOCA --t 600 [--split val] [--llm] [--no-memory]

Телеметрия (воспроизведение прогона, только чтение) → память о событии → генератор (6 каналов)
→ верификатор гипотез (все наблюдаемые каналы, окно + ранняя фаза) → проверка отклонения и новизны
→ итоговое правило → ответ AnalystAgent → верификатор ответа (ENFORCE) → решение политики.
Истинный сценарий берётся из меток только для подписи в выводе.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atomic.diagnostics.labels import LabelStore  # noqa: E402
from atomic.diagnostics.nppad import NPPADDataset  # noqa: E402
from atomic.diagnostics.telemetry import TelemetryReplay  # noqa: E402
from atomic.memory.vector import VectorMemory  # noqa: E402
from atomic.orchestration.workflow import AtomicWorkflow  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="LOCA")
    ap.add_argument("--t", type=float, default=600.0, help="момент от начала записи, с")
    ap.add_argument("--split", default="val", choices=["train", "val", "probe"])
    ap.add_argument("--run", type=int, default=0)
    ap.add_argument("--llm", action="store_true", help="добавить пояснение LLM (нужна Ollama)")
    ap.add_argument("--no-memory", action="store_true", help="без памяти о начале события")
    args = ap.parse_args()

    ds = NPPADDataset()
    labels = LabelStore.load(ds.directory)
    run = [r for r in ds.split(args.split) if labels.scenario(r) == args.scenario][args.run]
    lab = labels.get(run)
    wf = AtomicWorkflow(vector_memory=VectorMemory(), verification_mode="shadow")
    r = wf.diagnose(TelemetryReplay(ds, run, episode_memory=not args.no_memory), args.t, use_llm=args.llm)
    rep = wf.last_diagnosis
    d = r.bundle.decision
    print(f"[для подписи: истинный сценарий {lab.scenario}, тяжесть {lab.severity:g} %]")
    print(f"Исход диагностики: {rep.outcome.value}; память о событии: {'да' if rep.episode_memory else 'нет'}; "
          f"новизна: {rep.novelty.novel} (событие {rep.novelty.episode_score} / порог {rep.novelty.episode_threshold}, "
          f"окно {rep.novelty.window_score} / {rep.novelty.window_threshold})")
    print(f"Решение верификатора ответа: {d.effect.value} (пакет {d.applied_pack_id}, нарушено {list(d.violated_rule_ids)}, "
          f"уверенность {d.propagated_confidence:.2f})")
    print(f"Задержка, мс: {rep.latency_ms}")
    print("\n--- ответ оператору ---\n" + r.answer)


if __name__ == "__main__":
    main()
