#!/usr/bin/env python3
"""
Подготовка NPPAD одной командой (критерий готовности этапа A).

    uv run python scripts/load_nppad.py

Предварительно: сырые данные в NPPAD/raw (git clone репозитория thu-inet),
см. docs/datasets.md. Результат: data/processed/nppad/{runs.npz, manifest.json, labels.json}.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atomic.diagnostics.nppad import build_processed  # noqa: E402


def main() -> None:
    m = build_processed()
    print(f"Источник: {m['source']['repository']} @ {m['source']['commit']}")
    print(f"Прогонов: {m['n_runs']}, строк: {m['n_rows']}, каналов: {len(m['channels'])}")
    for name in ("train", "val", "test", "probe"):
        print(f"  {name:6s}: {len(m['split'][name])} прогонов")
    if m["dropped_columns"]:
        print(f"Отброшены столбцы вне реестра: {m['dropped_columns']}")
    if m["constant_channels_train"]:
        print(f"Постоянные на train каналы ({len(m['constant_channels_train'])}): {m['constant_channels_train']}")
    if m["issues"]:
        print(f"Замечания ({len(m['issues'])}): {m['issues'][:5]}")


if __name__ == "__main__":
    main()
