"""
Типы контура гипотез о состоянии АЭС (этапы A–B дорожной карты).

Поток: ObservationWindow (все каналы, только для верификатора)
      → барьер → RestrictedWindow (разрешённые каналы, без меток)
      → генератор → HypothesisSet (top-k, «вероятна ситуация X, уверенность Y»)
      → верификатор → HypothesisVerdict (свидетельства «за»/«против» + аналоги).

Инвариант «Upgrade to Guided Operations» (этап 2): ни один тип не содержит
рекомендаций действий — только интерпретацию с уверенностью и ссылками.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from atomic.contracts.verification import EvidenceKind, EvidenceRef


@dataclass(frozen=True, eq=False)
class ObservationWindow:
    """Окно наблюдений по ВСЕМ каналам. Генератору не передаётся.

    window_id и run_id непрозрачны (хэши): по ним нельзя восстановить сценарий.
    values: массив формы (T, C), нормализованные значения в порядке channels.
    raw_values: те же данные в физических единицах (для свидетельств).
    """

    window_id: str
    run_id: str
    t_start: float  # время от начала прогона, с (только для оценщика и трасс)
    t_end: float
    channels: tuple[str, ...]
    values: np.ndarray
    raw_values: np.ndarray

    def __post_init__(self) -> None:
        if self.values.shape != self.raw_values.shape:
            raise ValueError("values и raw_values должны иметь одну форму")
        if self.values.ndim != 2 or self.values.shape[1] != len(self.channels):
            raise ValueError("values должен иметь форму (T, len(channels))")

    def channel(self, name: str, raw: bool = False) -> np.ndarray:
        idx = self.channels.index(name)
        return (self.raw_values if raw else self.values)[:, idx]


@dataclass(frozen=True, eq=False)
class EpisodeContext:
    """Память о текущем событии для верификатора (D-030).

    baseline — состояние станции непосредственно перед началом события (z-значения, все каналы);
    early — наблюдения с начала события, но не дальше конца текущего окна (причинность)
    и не дальше горизонта ранней фазы; шаг 10 с, early[0] — момент начала.
    В NPPAD начало события = начало прогона (неисправность вводится в 0,5 с, точка 0 с —
    исходное стационарное состояние). В эксплуатации — момент первой сигнализации.
    """

    episode_id: str
    onset_window_id: str  # окно, начинающееся в момент начала события (ссылка для свидетельств)
    channels: tuple[str, ...]
    baseline: np.ndarray  # (C,)
    early: np.ndarray  # (n, C)
    step_s: float = 10.0
    baseline_raw: np.ndarray | None = None  # (C,) исходное состояние в физических единицах

    @property
    def horizon_s(self) -> float:
        return (len(self.early) - 1) * self.step_s


@dataclass(frozen=True, eq=False)
class RestrictedWindow:
    """Вход генератора: только разрешённые каналы, без run_id, времени и меток.

    Создаётся исключительно через atomic.diagnostics.barrier.restrict().
    """

    window_id: str
    channels: tuple[str, ...]
    values: np.ndarray  # (T, len(channels)), нормализованные значения

    def __post_init__(self) -> None:
        if self.values.ndim != 2 or self.values.shape[1] != len(self.channels):
            raise ValueError("values должен иметь форму (T, len(channels))")


@dataclass(frozen=True)
class ScenarioHypothesis:
    """«Вероятна ситуация X, уверенность Y» — без рекомендаций действий."""

    scenario: str  # код сценария NPPAD: LOCA, SGATR, ...
    confidence: float  # калиброванная вероятность, 0..1
    rank: int  # 1 = самая вероятная
    channels_used: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence вне [0, 1]: {self.confidence}")
        if self.rank < 1:
            raise ValueError("rank начинается с 1")

    def describe(self, scenario_titles: dict[str, str] | None = None) -> str:
        title = (scenario_titles or {}).get(self.scenario, self.scenario)
        return f"Вероятна ситуация {self.scenario} ({title}), уверенность {self.confidence:.0%}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "confidence": self.confidence,
            "rank": self.rank,
            "channels_used": list(self.channels_used),
        }


@dataclass(frozen=True)
class HypothesisSet:
    """Top-k гипотез по одному окну (защита от H3 и H4: несколько гипотез, калибровка)."""

    window_id: str
    hypotheses: tuple[ScenarioHypothesis, ...]
    generator_version: str

    def __post_init__(self) -> None:
        if not self.hypotheses:
            raise ValueError("HypothesisSet должен содержать хотя бы одну гипотезу")
        confs = [h.confidence for h in self.hypotheses]
        if any(a < b for a, b in zip(confs, confs[1:])):
            raise ValueError("гипотезы должны быть упорядочены по убыванию уверенности")
        if sum(confs) > 1.0 + 1e-6:
            raise ValueError(f"сумма уверенностей top-k > 1: {sum(confs):.4f}")
        if [h.rank for h in self.hypotheses] != list(range(1, len(self.hypotheses) + 1)):
            raise ValueError("ранги должны идти 1..k")

    @property
    def top(self) -> ScenarioHypothesis:
        return self.hypotheses[0]

    def scenarios(self) -> list[str]:
        return [h.scenario for h in self.hypotheses]

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "generator_version": self.generator_version,
            "hypotheses": [h.to_dict() for h in self.hypotheses],
        }


class EvidenceDirection(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"


@dataclass(frozen=True)
class ChannelEvidence:
    """Свидетельство по конкретному каналу в конкретном окне."""

    channel: str
    window_id: str
    direction: EvidenceDirection
    observed: float  # наблюдаемый признак (например, изменение за окно, z)
    expected: float  # ожидание по сигнатуре сценария
    weight: float  # значимость канала для сценария, 0..1
    detail: str = ""

    @property
    def ref_id(self) -> str:
        return f"{self.window_id}#{self.channel}"

    def to_evidence_ref(self, trust_base: float) -> EvidenceRef:
        return EvidenceRef(
            ref_id=self.ref_id,
            kind=EvidenceKind.TELEMETRY,
            trust_base=trust_base,
            locator=self.channel,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "window_id": self.window_id,
            "direction": self.direction.value,
            "observed": self.observed,
            "expected": self.expected,
            "weight": self.weight,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class EventAnalogue:
    """Аналог из базы исторических событий: «в аналогичных случаях наблюдалось…»."""

    event_id: str
    title: str
    summary: str
    score: float
    matched_terms: tuple[str, ...] = ()
    match_basis: str = "text"  # "initiating_event" — совпал размеченный тип исходного события ETHZ

    def describe(self) -> str:
        return f"В аналогичных случаях наблюдалось: {self.title} — {self.summary}"

    def to_evidence_ref(self, trust_base: float) -> EvidenceRef:
        return EvidenceRef(
            ref_id=self.event_id,
            kind=EvidenceKind.HISTORICAL_EVENT,
            trust_base=trust_base,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "title": self.title,
            "summary": self.summary,
            "score": self.score,
            "matched_terms": list(self.matched_terms),
            "match_basis": self.match_basis,
        }


@dataclass(frozen=True)
class NoveltyAssessment:
    """Есть ли отклонение от исходного состояния и похоже ли событие на известные сценарии (D-032)."""

    anomaly: bool  # хотя бы min_channels каналов отклонились от исходного состояния
    deviating_channels: tuple[str, ...]
    novel: bool  # событие не похоже ни на один известный сценарий
    episode_score: float | None  # расстояние до ближайшего известного события (по ранней фазе)
    episode_threshold: float
    window_score: float  # расстояние до ближайшего известного окна
    window_threshold: float
    nearest_scenario: str | None
    basis: str  # "episode+window" или "window"
    detector_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "anomaly": self.anomaly, "deviating_channels": list(self.deviating_channels), "novel": self.novel,
            "episode_score": self.episode_score, "episode_threshold": self.episode_threshold,
            "window_score": self.window_score, "window_threshold": self.window_threshold,
            "nearest_scenario": self.nearest_scenario, "basis": self.basis, "detector_version": self.detector_version,
        }


class VerdictStatus(str, Enum):
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class HypothesisVerdict:
    """Результат независимой проверки одной гипотезы по полному набору каналов."""

    hypothesis: ScenarioHypothesis
    status: VerdictStatus
    support_ratio: float  # взвешенная доля подтверждающих свидетельств, 0..1
    evidence_for: tuple[ChannelEvidence, ...] = ()
    evidence_against: tuple[ChannelEvidence, ...] = ()
    analogues: tuple[EventAnalogue, ...] = ()
    verifier_version: str = ""
    notes: tuple[str, ...] = field(default=())

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis": self.hypothesis.to_dict(),
            "status": self.status.value,
            "support_ratio": self.support_ratio,
            "evidence_for": [e.to_dict() for e in self.evidence_for],
            "evidence_against": [e.to_dict() for e in self.evidence_against],
            "analogues": [a.to_dict() for a in self.analogues],
            "verifier_version": self.verifier_version,
            "notes": list(self.notes),
        }


class DiagnosisOutcome(str, Enum):
    NO_ANOMALY = "no_anomaly"  # отклонений от исходного состояния не обнаружено
    NOVEL = "novel"  # событие не похоже ни на один известный сценарий
    UNCONFIRMED = "unconfirmed"  # ни одна гипотеза не подтверждена
    AMBIGUOUS = "ambiguous"  # подтверждено несколько гипотез с близкой поддержкой
    IDENTIFIED = "identified"  # одна гипотеза подтверждена и заметно сильнее остальных


@dataclass(frozen=True)
class DiagnosticReport:
    """Итог левого контура по одному окну: гипотезы, проверка, новизна, исход (D-033)."""

    window_id: str
    hypotheses: HypothesisSet
    verdicts: tuple[HypothesisVerdict, ...]
    novelty: NoveltyAssessment
    outcome: DiagnosisOutcome
    selected: ScenarioHypothesis | None
    candidates: tuple[str, ...]  # сценарии, которые остаются в рассмотрении
    episode_memory: bool
    versions: dict[str, str] = field(default_factory=dict)
    latency_ms: dict[str, float] = field(default_factory=dict)

    def verdict(self, scenario: str) -> HypothesisVerdict | None:
        return next((v for v in self.verdicts if v.hypothesis.scenario == scenario), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "hypotheses": self.hypotheses.to_dict(),
            "verdicts": [v.to_dict() for v in self.verdicts],
            "novelty": self.novelty.to_dict(),
            "outcome": self.outcome.value,
            "selected": self.selected.to_dict() if self.selected else None,
            "candidates": list(self.candidates),
            "episode_memory": self.episode_memory,
            "versions": dict(self.versions),
            "latency_ms": dict(self.latency_ms),
        }
