"""
Policy engine: результаты правил + уверенность → PolicyDecision.

Порядок: нарушено правило severity=error → BLOCK;
иначе нарушено warning-правило или уверенность ниже порога пакета → REQUIRE_HUMAN;
иначе ALLOW. Уверенность ответа = min силы утверждений, сила утверждения =
min trust_base его источников (проект 7-го семестра).
"""

from __future__ import annotations

from atomic.contracts.verification import Claim, PolicyDecision, PolicyEffect, RulePack


def propagate_confidence(claims: tuple[Claim, ...]) -> float:
    if not claims:
        return 0.0
    return min(c.strength() for c in claims)


def decide(pack: RulePack, rule_results: dict[str, bool], claims: tuple[Claim, ...]) -> PolicyDecision:
    violated = tuple(r.rule_id for r in pack.rules if not rule_results.get(r.rule_id, False))
    errors = [r for r in violated if pack.rule(r).severity == "error"]
    warnings = [r for r in violated if pack.rule(r).severity == "warning"]
    confidence = propagate_confidence(claims)
    notes: list[str] = []
    if errors:
        effect = PolicyEffect.BLOCK
        notes.append(f"нарушены правила уровня error: {', '.join(errors)}")
    elif warnings or confidence < pack.min_confidence:
        effect = PolicyEffect.REQUIRE_HUMAN
        if warnings:
            notes.append(f"нарушены правила уровня warning: {', '.join(warnings)}")
        if confidence < pack.min_confidence:
            notes.append(f"уверенность {confidence:.2f} ниже порога {pack.min_confidence:.2f}")
    else:
        effect = PolicyEffect.ALLOW
        notes.append("все правила выполнены")
    return PolicyDecision(
        effect=effect,
        applied_pack_id=pack.pack_id,
        violated_rule_ids=violated,
        propagated_confidence=confidence,
        audit_notes=tuple(notes),
    )
