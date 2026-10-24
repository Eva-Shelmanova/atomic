"""
Подбор аналогов из базы исторических событий ETHZ (этап B2, требование 7).

Для гипотезы «вероятна ситуация X» ищутся события, в которых наблюдалось
похожее: BM25 по описанию, системе-источнику и причине события с запросом
из config/analogues.toml, плюс усиление при совпадении системы-источника.
Результат — EventAnalogue в форме «в аналогичных случаях наблюдалось…»,
без предписания действий (этап 3 «Passive Guidance» документа Upgrade, пассивный элемент).
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

from atomic.contracts.hypothesis import EventAnalogue
from atomic.diagnostics.config import PROJECT_ROOT, DiagnosticsConfig, default_config
from atomic.memory.bm25 import BM25

DEFAULT_MAPPING = PROJECT_ROOT / "config" / "analogues.toml"


def _summary(ev: dict[str, Any], limit: int = 220) -> str:
    text = " ".join((ev.get("description") or "").split())
    return text if len(text) <= limit else text[: limit - 1].rsplit(" ", 1)[0] + "…"


def _title(ev: dict[str, Any]) -> str:
    parts = [ev.get("site") or "?", ev.get("country") or "", (ev.get("date") or "")[:4]]
    origin = ev.get("origin")
    head = ", ".join(p for p in parts if p)
    return f"{head}: {origin}" if origin else head


class AnalogueFinder:
    def __init__(
        self,
        events: list[dict[str, Any]],
        mapping: dict[str, Any],
        source_version: str = "",
    ):
        self.events = events
        self.mapping = mapping
        self.source_version = source_version
        self._ids = {e["event_id"] for e in events}
        self._index = BM25(
            [" ".join(filter(None, (e.get("description"), e.get("origin"), e.get("cause")))) for e in events]
        )

    @classmethod
    def load(cls, config: DiagnosticsConfig | None = None, mapping_path: Path | None = None) -> AnalogueFinder:
        cfg = config or default_config()
        path = cfg.resolve("ethz_events")
        if not path.exists():
            raise FileNotFoundError(f"{path} не найден. Подготовьте базу: uv run python scripts/load_ethz.py")
        payload = json.loads(path.read_text(encoding="utf-8"))
        with (mapping_path or DEFAULT_MAPPING).open("rb") as fh:
            mapping = tomllib.load(fh)
        return cls(payload["events"], mapping, source_version=payload.get("source_sha256", ""))

    @property
    def version(self) -> str:
        return f"analogues-{self.mapping.get('version', '0')}+ethz-{self.source_version or 'dev'}"

    def has_event(self, event_id: str) -> bool:
        return event_id in self._ids

    def find(self, scenario: str, k: int = 3) -> list[EventAnalogue]:
        spec = self.mapping.get("scenario", {}).get(scenario)
        if not spec or not spec.get("query"):
            return []
        query = spec["query"]
        origins = {o.lower() for o in spec.get("origin", [])}
        boost = float(self.mapping.get("origin_boost", 1.0))
        min_score = float(self.mapping.get("min_score", 0.0))
        scored = []
        for i, s in enumerate(self._index.scores(query)):
            if s <= 0:
                continue
            ev = self.events[i]
            if origins and (ev.get("origin") or "").lower() in origins:
                s *= boost
            if s >= min_score:
                scored.append((s, i))
        scored.sort(key=lambda x: (-x[0], x[1]))
        out = []
        seen: set[tuple] = set()
        for s, i in scored:
            if len(out) == k:
                break
            ev = self.events[i]
            key = (ev.get("site"), ev.get("date"), (ev.get("description") or "")[:80])
            if key in seen:  # одно событие на нескольких энергоблоках записано несколькими строками
                continue
            seen.add(key)
            terms = [t for t in self._index.matched_terms(query, i) if "_" in t] or self._index.matched_terms(query, i)
            out.append(
                EventAnalogue(
                    event_id=ev["event_id"],
                    title=_title(ev),
                    summary=_summary(ev),
                    score=round(float(s), 3),
                    matched_terms=tuple(terms[:8]),
                )
            )
        return out
