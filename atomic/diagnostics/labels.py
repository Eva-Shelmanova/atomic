"""
Хранилище меток сценариев (ground truth), отделённое от данных.

Метки читает только подготовка данных (разбиение), обучение генератора
и офлайн-оценщик. В рабочем пути «окно → гипотеза → проверка» модуль
не импортируется; это проверяет контрактный тест барьера.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunLabel:
    run_id: str
    scenario: str  # код класса NPPAD: LOCA, SGATR, ...
    severity: float | None  # тяжесть, % (смысл зависит от класса, см. README Table 1)
    source_file: str  # путь относительно каталога NPPAD (содержит имя сценария!)
    malfunction: int | None = None  # номер неисправности PCTRAN из Transient Report


class LabelStore:
    FILENAME = "labels.json"

    def __init__(self, labels: dict[str, RunLabel]):
        self._labels = labels

    def __len__(self) -> int:
        return len(self._labels)

    def __contains__(self, run_id: str) -> bool:
        return run_id in self._labels

    def get(self, run_id: str) -> RunLabel:
        return self._labels[run_id]

    def scenario(self, run_id: str) -> str:
        return self._labels[run_id].scenario

    def run_ids(self, scenario: str | None = None) -> list[str]:
        return sorted(r for r, lab in self._labels.items() if scenario is None or lab.scenario == scenario)

    def save(self, directory: Path) -> Path:
        path = directory / self.FILENAME
        payload = {r: asdict(lab) for r, lab in sorted(self._labels.items())}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        return path

    @classmethod
    def load(cls, directory: Path) -> LabelStore:
        raw = json.loads((directory / cls.FILENAME).read_text(encoding="utf-8"))
        return cls({r: RunLabel(**d) for r, d in raw.items()})
