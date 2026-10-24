"""
Справка по сценариям NPPAD в базе знаний ассистента (D-043).

Файлы dataset/scenarios/<код>.txt строит scripts/build_scenario_kb.py из журналов симулятора
PCTRAN обучающих прогонов. Каждый файл добавляется в векторную память одним документом с
метаданными {"source", "scenario", "kind"}: по ним диагностический ответ ссылается только на
справку установленного сценария ([D…], D-040).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

SCENARIO_KB_DIR = Path(__file__).resolve().parents[2] / "dataset" / "scenarios"


class _VectorMemoryLike(Protocol):
    def has_content(self, content: str) -> bool: ...
    def add(self, content: str, metadata: dict | None = None) -> None: ...


def scenario_documents(root: Path = SCENARIO_KB_DIR) -> list[tuple[str, dict[str, Any]]]:
    return [
        (p.read_text(encoding="utf-8").strip(),
         {"source": f"dataset/scenarios/{p.name}", "scenario": p.stem, "kind": "scenario_reference"})
        for p in sorted(root.glob("*.txt"))
    ]


def seed_scenario_kb(memory: _VectorMemoryLike, root: Path = SCENARIO_KB_DIR) -> int:
    """Добавляет справку в память (повторный вызов не создаёт дубликатов). Возвращает число добавленных."""
    added = 0
    for content, meta in scenario_documents(root):
        if content and not memory.has_content(content):
            memory.add(content, metadata=meta)
            added += 1
    return added


def find_scenario_docs(memory, scenario: str, title: str = "", min_score: float = 0.6, top_k: int = 5) -> list[dict]:
    """Фрагменты базы знаний для установленного сценария (D-040, D-043).

    Отбор: у pgvector — косинусная близость не ниже min_score, у BM25 — ненулевая оценка;
    документы, размеченные другим сценарием, отбрасываются. Недоступная база знаний → [].
    """
    if memory is None:
        return []
    try:
        found = memory.search(f"{scenario} {title}".strip(), top_k=top_k)
    except Exception:  # база знаний недоступна — диагностический ответ без справочных материалов
        return []
    return [d for d in found if d.get("score", 1.0) >= min_score
            and (d.get("metadata") or {}).get("scenario", scenario) == scenario]
