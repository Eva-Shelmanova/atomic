"""
AnswerVerifier: AgentResponse → VerificationBundle.

Агенты кладут утверждения в AgentResponse.metadata["verification"]:
    {"claims": [{"claim_id": ..., "text": ..., "evidence": [{"ref_id", "kind", "locator"}]}]}
Верификатор назначает trust_base по виду источника из пакета правил (D-015),
применяет правила и вызывает policy engine. LLM не участвует.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Any

from atomic.contracts.verification import (
    Claim,
    EvidenceKind,
    EvidenceRef,
    PolicyEffect,
    RulePack,
    VerificationBundle,
)
from atomic.verification.policy import decide
from atomic.verification.resolver import EvidenceResolver
from atomic.verification.rulepack import load_rulepack
from atomic.verification.rules import RULES


class VerificationMode(str, Enum):
    OFF = "off"  # проверка не выполняется (поведение 6-го семестра)
    SHADOW = "shadow"  # решение считается и пишется в трассу, текст не меняется
    ENFORCE = "enforce"  # решение применяется к выпуску ответа

    @classmethod
    def from_env(cls) -> VerificationMode:
        return cls(os.getenv("ATOMIC_VERIFICATION_MODE", cls.SHADOW.value).strip().lower())


class AnswerVerifier:
    def __init__(self, pack: RulePack | None = None, resolver: EvidenceResolver | None = None):
        self.pack = pack or load_rulepack()
        self.resolver = resolver or EvidenceResolver()
        unknown = [r.rule_id for r in self.pack.rules if r.rule_id not in RULES]
        if unknown:
            raise ValueError(f"в пакете {self.pack.pack_id} правила без реализации: {unknown}")

    def _trust(self, kind: EvidenceKind) -> float:
        return float(self.pack.params["trust_base"][kind])

    def parse_claims(self, metadata: dict[str, Any] | None) -> tuple[Claim, ...]:
        payload = (metadata or {}).get("verification") or {}
        claims = []
        for i, c in enumerate(payload.get("claims", []), start=1):
            evidence = []
            for e in c.get("evidence", []):
                kind = EvidenceKind(e["kind"])
                evidence.append(
                    EvidenceRef(ref_id=str(e["ref_id"]), kind=kind, trust_base=self._trust(kind), locator=e.get("locator"))
                )
            claims.append(Claim(claim_id=str(c.get("claim_id") or f"c{i}"), text=str(c.get("text", "")),
                                evidence=tuple(evidence), attributes=dict(c.get("attributes", {}))))
        return tuple(claims)

    def verify_claims(self, query: str, claims: tuple[Claim, ...], agent: str | None = None) -> VerificationBundle:
        results: dict[str, bool] = {}
        details: dict[str, list[str]] = {}
        for rule in self.pack.rules:
            passed, reasons = RULES[rule.rule_id](claims, self.resolver)
            results[rule.rule_id] = passed
            details[rule.rule_id] = reasons
        decision = decide(self.pack, results, claims)
        return VerificationBundle(
            query=query, claims=claims, rule_results=results, decision=decision, agent=agent, rule_details=details
        )

    def verify(self, query: str, response: Any) -> VerificationBundle:
        """response — AgentResponse (или объект с полями metadata и agent_type)."""
        agent_type = getattr(response, "agent_type", None)
        agent = getattr(agent_type, "value", agent_type)
        return self.verify_claims(query, self.parse_claims(getattr(response, "metadata", None)), agent=agent)


BLOCK_TEMPLATE = (
    "Ответ не выпущен: он не прошёл детерминированную проверку (пакет {pack}; нарушены правила: {rules}). "
    "Обратитесь к первоисточникам или к эксперту."
)
HUMAN_TEMPLATE = "[Требуется оценка оператора: {notes}. Уверенность {conf:.0%}.]\n\n{content}"


def release_text(content: str, bundle: VerificationBundle) -> str:
    """Текст, который выпускается оператору в режиме ENFORCE."""
    d = bundle.decision
    if d.effect is PolicyEffect.BLOCK:
        return BLOCK_TEMPLATE.format(pack=d.applied_pack_id, rules=", ".join(d.violated_rule_ids))
    if d.effect is PolicyEffect.REQUIRE_HUMAN:
        return HUMAN_TEMPLATE.format(notes="; ".join(d.audit_notes), conf=d.propagated_confidence, content=content)
    return content
