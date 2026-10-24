"""
Верификатор ответа (этап B3, требование 9).

Claim → EvidenceRef → RulePack → PolicyDecision. Модуль стоит между
AtomicWorkflow._route_and_process() и записью ответа в память; это не ещё
один агент в цепочке, а детерминированный шлюз выпуска.
"""

from atomic.verification.policy import decide
from atomic.verification.resolver import EvidenceResolver
from atomic.verification.rulepack import load_rulepack
from atomic.verification.verifier import AnswerVerifier, VerificationMode, release_text

__all__ = ["AnswerVerifier", "EvidenceResolver", "VerificationMode", "decide", "load_rulepack", "release_text"]
