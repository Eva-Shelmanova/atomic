"""
Типы верификатора ответа.

Источник: проект «7-й семестр» (замечание о Verifier и минимальная
машиночитаемая схема). Смысл полей сохранён без изменений; добавлены
только проверки инвариантов и сериализация to_dict/from_dict для трасс
Langfuse и JSON-артефактов.

Идея: ответ = утверждения (Claim) + ссылки на артефакты (EvidenceRef)
+ проверка версионируемым набором правил (RulePack)
+ явное решение политики (PolicyDecision). LLM в решении не участвует.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EvidenceKind(str, Enum):
    """Происхождение факта — для trust, а не «уверенность модели»."""

    REGULATORY_DOC = "regulatory_doc"  # нормативный документ / процедура с id
    RAG_CHUNK = "rag_chunk"  # фрагмент из базы знаний (векторная БД)
    TOOL_OUTPUT = "tool_output"  # вывод инструмента: код, расчёт, симулятор
    TELEMETRY = "telemetry"  # окно наблюдений по каналам (NPPAD)
    HISTORICAL_EVENT = "historical_event"  # событие из базы ETHZ
    HUMAN_ATTESTATION = "human"  # явное подтверждение человека
    MODEL_GENERATED = "model"  # только текст модели — самый низкий слой


@dataclass(frozen=True)
class EvidenceRef:
    """Ссылка на артефакт, а не на «место в промпте»."""

    ref_id: str  # стабильный id: chunk_id, window_id, event_id, run_id
    kind: EvidenceKind
    trust_base: float  # 0..1, базовый вес класса источника (задаёт политика домена)
    locator: str | None = None  # опционально: канал, страница, параграф, URI

    def __post_init__(self) -> None:
        if not self.ref_id:
            raise ValueError("EvidenceRef.ref_id не может быть пустым")
        if not 0.0 <= self.trust_base <= 1.0:
            raise ValueError(f"trust_base вне [0, 1]: {self.trust_base}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "ref_id": self.ref_id,
            "kind": self.kind.value,
            "trust_base": self.trust_base,
            "locator": self.locator,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvidenceRef:
        return cls(
            ref_id=str(d["ref_id"]),
            kind=EvidenceKind(d["kind"]),
            trust_base=float(d["trust_base"]),
            locator=d.get("locator"),
        )


@dataclass(frozen=True)
class Claim:
    """Одно проверяемое утверждение в ответе пользователю."""

    claim_id: str
    text: str  # формулировка, которую показываем и проверяем
    evidence: tuple[EvidenceRef, ...] = ()  # пустой tuple = «нет опоры» → правила режут

    def strength(self) -> float:
        """Сила утверждения = min trust_base по его свидетельствам (0, если их нет)."""
        if not self.evidence:
            return 0.0
        return min(e.trust_base for e in self.evidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "evidence": [e.to_dict() for e in self.evidence],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Claim:
        return cls(
            claim_id=str(d["claim_id"]),
            text=str(d.get("text", "")),
            evidence=tuple(EvidenceRef.from_dict(e) for e in d.get("evidence", [])),
        )


@dataclass(frozen=True)
class Rule:
    """Одна детерминированная проверка (не LLM). Реализация — по rule_id в реестре."""

    rule_id: str
    severity: str  # "error" | "warning" — как политика трактует нарушение
    description: str = ""

    def __post_init__(self) -> None:
        if self.severity not in ("error", "warning"):
            raise ValueError(f"severity должен быть error|warning, получено {self.severity}")


@dataclass(frozen=True)
class RulePack:
    """Версионируемый набор правил (для аудита: какой пакет применили)."""

    pack_id: str  # например "aes-core-1.0.0"
    rules: tuple[Rule, ...]
    min_confidence: float = 0.5  # ниже — REQUIRE_HUMAN
    params: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)

    def rule(self, rule_id: str) -> Rule:
        for r in self.rules:
            if r.rule_id == rule_id:
                return r
        raise KeyError(rule_id)


class PolicyEffect(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"  # ответ не выпускается, оператор видит заглушку
    REQUIRE_HUMAN = "require_human"  # выпуск только с пометкой / после оценки человеком


@dataclass(frozen=True)
class PolicyDecision:
    """Результат policy engine и агрегация нарушений."""

    effect: PolicyEffect
    applied_pack_id: str
    violated_rule_ids: tuple[str, ...] = ()
    propagated_confidence: float = 0.0  # 0..1 после агрегации по цепочке claim → evidence
    audit_notes: tuple[str, ...] = ()  # короткие машинные причины

    def to_dict(self) -> dict[str, Any]:
        return {
            "effect": self.effect.value,
            "applied_pack_id": self.applied_pack_id,
            "violated_rule_ids": list(self.violated_rule_ids),
            "propagated_confidence": self.propagated_confidence,
            "audit_notes": list(self.audit_notes),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PolicyDecision:
        return cls(
            effect=PolicyEffect(d["effect"]),
            applied_pack_id=str(d["applied_pack_id"]),
            violated_rule_ids=tuple(d.get("violated_rule_ids", [])),
            propagated_confidence=float(d.get("propagated_confidence", 0.0)),
            audit_notes=tuple(d.get("audit_notes", [])),
        )


@dataclass
class VerificationBundle:
    """Всё, что нужно для аудита одного ask(): запрос, утверждения, правила, решение."""

    query: str
    claims: tuple[Claim, ...]
    rule_results: dict[str, bool]  # rule_id -> pass
    decision: PolicyDecision
    agent: str | None = None
    rule_details: dict[str, list[str]] = field(default_factory=dict)  # rule_id -> причины

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "agent": self.agent,
            "claims": [c.to_dict() for c in self.claims],
            "rule_results": dict(self.rule_results),
            "rule_details": {k: list(v) for k, v in self.rule_details.items()},
            "decision": self.decision.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> VerificationBundle:
        return cls(
            query=str(d["query"]),
            agent=d.get("agent"),
            claims=tuple(Claim.from_dict(c) for c in d.get("claims", [])),
            rule_results={k: bool(v) for k, v in d.get("rule_results", {}).items()},
            rule_details={k: list(v) for k, v in d.get("rule_details", {}).items()},
            decision=PolicyDecision.from_dict(d["decision"]),
        )
