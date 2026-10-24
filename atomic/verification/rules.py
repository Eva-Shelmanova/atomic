"""
Детерминированные правила пакета aes-core.

Каждое правило — чистая функция (claims, resolver) → (прошло, причины).
Реестр RULES сопоставляет rule_id из пакета с реализацией.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from atomic.contracts.verification import Claim, EvidenceKind
from atomic.verification.resolver import EvidenceResolver

RuleResult = tuple[bool, list[str]]
RuleFn = Callable[[tuple[Claim, ...], EvidenceResolver], RuleResult]


def evidence_required(claims: tuple[Claim, ...], resolver: EvidenceResolver) -> RuleResult:
    if not claims:
        return False, ["ответ не содержит утверждений (нет metadata['verification'])"]
    bad = [f"{c.claim_id}: нет ссылок на источники" for c in claims if not c.evidence]
    return not bad, bad


def no_model_only(claims: tuple[Claim, ...], resolver: EvidenceResolver) -> RuleResult:
    bad = [
        f"{c.claim_id}: опирается только на текст модели"
        for c in claims
        if c.evidence and all(e.kind is EvidenceKind.MODEL_GENERATED for e in c.evidence)
    ]
    return not bad, bad


def ref_resolvable(claims: tuple[Claim, ...], resolver: EvidenceResolver) -> RuleResult:
    bad = [
        f"{c.claim_id}: ссылка {e.kind.value}:{e.ref_id} не разрешается"
        for c in claims
        for e in c.evidence
        if not resolver.resolves(e)
    ]
    return not bad, bad


DEFINITE_OUTCOMES = frozenset({"identified", "no_anomaly"})


def diagnosis_confirmed(claims: tuple[Claim, ...], resolver: EvidenceResolver) -> RuleResult:
    """Диагностическое утверждение выпускается без пометки, только если исход определённый:
    ситуация установлена однозначно (identified) или отклонений нет (no_anomaly)."""
    bad = [
        f"{c.claim_id}: исход диагностики «{c.attributes.get('outcome')}» — ситуация не установлена однозначно"
        for c in claims
        if c.attributes.get("type") == "diagnosis" and c.attributes.get("outcome") not in DEFINITE_OUTCOMES
    ]
    return not bad, bad


# Указания к действию недопустимы (инварианты «Upgrade to Guided Operations»: этапы 2–3 без предписаний).
DIRECTIVE_PATTERNS = re.compile(
    r"\b(выполните|откройте|закройте|включите|отключите|увеличьте|уменьшите|переведите|запустите|остановите|"
    r"подайте|снизьте|повысьте|сбросьте|необходимо|следует|рекомендуется|рекомендуем|нужно\s+(?:срочно\s+)?"
    r"(?:открыть|закрыть|включить|отключить|снизить|повысить|запустить|остановить))\b",
    re.IGNORECASE,
)


def no_directives(claims: tuple[Claim, ...], resolver: EvidenceResolver) -> RuleResult:
    bad = []
    for c in claims:
        m = DIRECTIVE_PATTERNS.search(c.text or "")
        if m:
            bad.append(f"{c.claim_id}: указание к действию «{m.group(0)}»")
    return not bad, bad


RULES: dict[str, RuleFn] = {
    "EVIDENCE_REQUIRED": evidence_required,
    "NO_MODEL_ONLY": no_model_only,
    "REF_RESOLVABLE": ref_resolvable,
    "DIAGNOSIS_CONFIRMED": diagnosis_confirmed,
    "NO_DIRECTIVES": no_directives,
}
