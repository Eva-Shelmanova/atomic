"""
Семантическая модель (граф знаний) диагностической информации.

Сущности и связи заданы дорожной картой (этап A): канал, окно наблюдений,
сценарий, историческое событие, документ и его фрагмент, гипотеза,
свидетельство, утверждение, правило, решение политики. У каждой сущности
есть идентификатор и происхождение (Provenance).

Граф хранится в памяти и сериализуется в JSON; схема связей проверяется
при добавлении ребра, поэтому некорректная связь (например, правило →
канал) не может попасть в граф. Отдельная графовая СУБД не используется (D-009).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class EntityType(str, Enum):
    CHANNEL = "channel"  # измерительный канал (P, TAVG, ...)
    WINDOW = "window"  # окно наблюдений прогона
    RUN = "run"  # прогон симулятора (непрозрачный id)
    SCENARIO = "scenario"  # класс ситуации NPPAD (LOCA, SGATR, ...)
    HISTORICAL_EVENT = "historical_event"  # событие из базы ETHZ
    DOCUMENT = "document"  # документ базы знаний (инструкция, регламент)
    FRAGMENT = "fragment"  # фрагмент (чанк) документа
    HYPOTHESIS = "hypothesis"  # гипотеза о состоянии
    EVIDENCE = "evidence"  # свидетельство «за» или «против»
    CLAIM = "claim"  # утверждение в ответе оператору
    RULE = "rule"  # детерминированное правило проверки
    RULE_PACK = "rule_pack"  # версионируемый набор правил
    POLICY_DECISION = "policy_decision"  # решение ALLOW / REQUIRE_HUMAN / BLOCK


class RelationType(str, Enum):
    PART_OF = "part_of"  # окно → прогон, фрагмент → документ, правило → пакет
    OBSERVES = "observes"  # окно → канал
    ABOUT = "about"  # гипотеза → окно
    PROPOSES = "proposes"  # гипотеза → сценарий
    REFERS_TO = "refers_to"  # свидетельство → канал / окно / событие / фрагмент
    SUPPORTS = "supports"  # свидетельство → гипотеза или утверждение
    CONTRADICTS = "contradicts"  # свидетельство → гипотеза
    ANALOGOUS_TO = "analogous_to"  # историческое событие → сценарий
    SIGNATURE_OF = "signature_of"  # канал → сценарий (канал входит в сигнатуру)
    CHECKED_BY = "checked_by"  # утверждение → правило
    DECIDES = "decides"  # решение → утверждение
    APPLIES = "applies"  # решение → пакет правил


E, R = EntityType, RelationType

# Допустимые пары (тип источника, тип цели) для каждой связи.
RELATION_SCHEMA: dict[RelationType, frozenset[tuple[EntityType, EntityType]]] = {
    R.PART_OF: frozenset({(E.WINDOW, E.RUN), (E.FRAGMENT, E.DOCUMENT), (E.RULE, E.RULE_PACK)}),
    R.OBSERVES: frozenset({(E.WINDOW, E.CHANNEL)}),
    R.ABOUT: frozenset({(E.HYPOTHESIS, E.WINDOW)}),
    R.PROPOSES: frozenset({(E.HYPOTHESIS, E.SCENARIO)}),
    R.REFERS_TO: frozenset(
        {
            (E.EVIDENCE, E.CHANNEL),
            (E.EVIDENCE, E.WINDOW),
            (E.EVIDENCE, E.HISTORICAL_EVENT),
            (E.EVIDENCE, E.FRAGMENT),
        }
    ),
    R.SUPPORTS: frozenset({(E.EVIDENCE, E.HYPOTHESIS), (E.EVIDENCE, E.CLAIM)}),
    R.CONTRADICTS: frozenset({(E.EVIDENCE, E.HYPOTHESIS)}),
    R.ANALOGOUS_TO: frozenset({(E.HISTORICAL_EVENT, E.SCENARIO)}),
    R.SIGNATURE_OF: frozenset({(E.CHANNEL, E.SCENARIO)}),
    R.CHECKED_BY: frozenset({(E.CLAIM, E.RULE)}),
    R.DECIDES: frozenset({(E.POLICY_DECISION, E.CLAIM)}),
    R.APPLIES: frozenset({(E.POLICY_DECISION, E.RULE_PACK)}),
}


@dataclass(frozen=True)
class Provenance:
    """Происхождение сущности: откуда она взялась и в какой версии."""

    source: str  # например "nppad@a8d54ed", "generator/1.0.0", "ethz-2020"
    version: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    def to_dict(self) -> dict[str, str]:
        return {"source": self.source, "version": self.version, "created_at": self.created_at}


@dataclass(frozen=True)
class Node:
    id: str
    type: EntityType
    provenance: Provenance
    attrs: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "type": self.type.value, "provenance": self.provenance.to_dict(), "attrs": self.attrs}


@dataclass(frozen=True)
class Edge:
    source: str
    relation: RelationType
    target: str
    attrs: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "relation": self.relation.value, "target": self.target, "attrs": self.attrs}


class KnowledgeGraph:
    """Граф знаний в памяти с проверкой схемы связей."""

    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}
        self._edges: list[Edge] = []

    # --- узлы ---
    def add_node(self, node: Node) -> Node:
        existing = self._nodes.get(node.id)
        if existing is not None and existing.type != node.type:
            raise ValueError(f"узел {node.id} уже существует с типом {existing.type.value}")
        self._nodes[node.id] = node
        return node

    def add(self, id: str, type: EntityType, provenance: Provenance, **attrs: Any) -> Node:
        return self.add_node(Node(id=id, type=type, provenance=provenance, attrs=attrs))

    def node(self, id: str) -> Node:
        return self._nodes[id]

    def has(self, id: str) -> bool:
        return id in self._nodes

    def nodes(self, type: EntityType | None = None) -> list[Node]:
        return [n for n in self._nodes.values() if type is None or n.type == type]

    # --- рёбра ---
    def link(self, source: str, relation: RelationType, target: str, **attrs: Any) -> Edge:
        if source not in self._nodes or target not in self._nodes:
            raise KeyError(f"нет узла для связи {source} -{relation.value}-> {target}")
        pair = (self._nodes[source].type, self._nodes[target].type)
        if pair not in RELATION_SCHEMA[relation]:
            raise ValueError(
                f"связь {relation.value} недопустима для {pair[0].value} → {pair[1].value}"
            )
        edge = Edge(source=source, relation=relation, target=target, attrs=attrs)
        self._edges.append(edge)
        return edge

    def edges(self, relation: RelationType | None = None) -> list[Edge]:
        return [e for e in self._edges if relation is None or e.relation == relation]

    def neighbors(self, id: str, relation: RelationType | None = None) -> list[Node]:
        return [
            self._nodes[e.target]
            for e in self._edges
            if e.source == id and (relation is None or e.relation == relation)
        ]

    def incoming(self, id: str, relation: RelationType | None = None) -> list[Node]:
        return [
            self._nodes[e.source]
            for e in self._edges
            if e.target == id and (relation is None or e.relation == relation)
        ]

    # --- сериализация ---
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "atomic-kg/1",
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "edges": [e.to_dict() for e in self._edges],
        }

    def to_json(self, **kw: Any) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, **kw)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> KnowledgeGraph:
        g = cls()
        for n in d["nodes"]:
            p = n["provenance"]
            g.add_node(
                Node(
                    id=n["id"],
                    type=EntityType(n["type"]),
                    provenance=Provenance(source=p["source"], version=p.get("version", ""), created_at=p["created_at"]),
                    attrs=n.get("attrs", {}),
                )
            )
        for e in d["edges"]:
            g.link(e["source"], RelationType(e["relation"]), e["target"], **e.get("attrs", {}))
        return g

    def stats(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for n in self._nodes.values():
            out[n.type.value] = out.get(n.type.value, 0) + 1
        out["edges"] = len(self._edges)
        return out


def schema_mermaid() -> str:
    """Диаграмма схемы (типы сущностей и допустимые связи) для документации."""
    lines = ["flowchart LR"]
    for rel, pairs in RELATION_SCHEMA.items():
        for src, dst in sorted(pairs, key=lambda p: (p[0].value, p[1].value)):
            lines.append(f"  {src.value} -- {rel.value} --> {dst.value}")
    return "\n".join(lines)
