"""
Детерминированные правила пакета aes-core.

Каждое правило — чистая функция (claims, resolver) → (прошло, причины).
Реестр RULES сопоставляет rule_id из пакета с реализацией.
"""

from __future__ import annotations

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


RULES: dict[str, RuleFn] = {
    "EVIDENCE_REQUIRED": evidence_required,
    "NO_MODEL_ONLY": no_model_only,
    "REF_RESOLVABLE": ref_resolvable,
}
