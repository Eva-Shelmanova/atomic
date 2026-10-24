"""
Контракты данных дипломной системы (этап A дорожной карты).

verification — типы верификатора ответа (проект 7-го семестра):
    EvidenceKind, EvidenceRef, Claim, Rule, RulePack, PolicyEffect,
    PolicyDecision, VerificationBundle.
hypothesis — типы контура гипотез:
    ObservationWindow, RestrictedWindow, ScenarioHypothesis, HypothesisSet,
    ChannelEvidence, EventAnalogue, HypothesisVerdict.
"""

from atomic.contracts.hypothesis import (
    ChannelEvidence,
    DiagnosisOutcome,
    DiagnosticReport,
    EpisodeContext,
    EvidenceDirection,
    EventAnalogue,
    HypothesisSet,
    HypothesisVerdict,
    NoveltyAssessment,
    ObservationWindow,
    RestrictedWindow,
    ScenarioHypothesis,
    VerdictStatus,
)
from atomic.contracts.verification import (
    Claim,
    EvidenceKind,
    EvidenceRef,
    PolicyDecision,
    PolicyEffect,
    Rule,
    RulePack,
    VerificationBundle,
)

__all__ = [
    "ChannelEvidence",
    "Claim",
    "DiagnosisOutcome",
    "DiagnosticReport",
    "EpisodeContext",
    "EvidenceDirection",
    "EvidenceKind",
    "EvidenceRef",
    "EventAnalogue",
    "HypothesisSet",
    "HypothesisVerdict",
    "NoveltyAssessment",
    "ObservationWindow",
    "PolicyDecision",
    "PolicyEffect",
    "RestrictedWindow",
    "Rule",
    "RulePack",
    "ScenarioHypothesis",
    "VerdictStatus",
    "VerificationBundle",
]
