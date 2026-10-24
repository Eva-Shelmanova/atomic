"""
Диагностический конвейер — левый контур дипломной системы (этап C).

    окно + память о событии
      → проверка отклонения (AnomalyGate)
      → барьер → генератор (top-k по 6 каналам)
      → верификатор гипотез (все наблюдаемые каналы, окно + ранняя фаза)
      → проверка новизны
      → итоговое правило выбора → DiagnosticReport

Итоговое правило (D-033):
  нет отклонения → NO_ANOMALY; событие новое → NOVEL;
  нет подтверждённых гипотез → UNCONFIRMED;
  одна подтверждённая или лидер опережает вторую подтверждённую ≥ margin → IDENTIFIED;
  иначе → AMBIGUOUS.
Система только читает данные и не формирует рекомендаций действий.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from atomic.contracts.hypothesis import (
    DiagnosisOutcome,
    DiagnosticReport,
    EpisodeContext,
    HypothesisVerdict,
    NoveltyAssessment,
    ObservationWindow,
    VerdictStatus,
)
from atomic.diagnostics.analogues import AnalogueFinder
from atomic.diagnostics.barrier import InformationBarrier
from atomic.diagnostics.channels import default_registry
from atomic.diagnostics.config import DiagnosticsConfig, default_config
from atomic.diagnostics.features import batch_features
from atomic.diagnostics.generator import HypothesisGenerator
from atomic.diagnostics.hypothesis_verifier import HypothesisVerifier
from atomic.diagnostics.novelty import AnomalyGate, NoveltyDetector
from atomic.diagnostics.onset import onset_track, stationary_points
from atomic.diagnostics.signatures import ScenarioSignatures


def select(verdicts: list[HypothesisVerdict], novelty: NoveltyAssessment, margin: float):
    """Итоговое правило выбора. Возвращает (исход, выбранная гипотеза, кандидаты)."""
    ranked = sorted(verdicts, key=lambda v: (-v.support_ratio, -v.hypothesis.confidence))
    if not novelty.anomaly:
        return DiagnosisOutcome.NO_ANOMALY, None, ()
    if novelty.novel:
        return DiagnosisOutcome.NOVEL, None, tuple(v.hypothesis.scenario for v in ranked)
    confirmed = [v for v in ranked if v.status is VerdictStatus.CONFIRMED]
    if not confirmed:
        return DiagnosisOutcome.UNCONFIRMED, None, tuple(v.hypothesis.scenario for v in ranked if v.status is not VerdictStatus.REJECTED)
    if len(confirmed) == 1 or confirmed[0].support_ratio - confirmed[1].support_ratio >= margin:
        return DiagnosisOutcome.IDENTIFIED, confirmed[0].hypothesis, (confirmed[0].hypothesis.scenario,)
    close = [v for v in confirmed if confirmed[0].support_ratio - v.support_ratio < margin]
    return DiagnosisOutcome.AMBIGUOUS, None, tuple(v.hypothesis.scenario for v in close)


@dataclass
class DiagnosticPipeline:
    generator: HypothesisGenerator
    verifier: HypothesisVerifier
    novelty: NoveltyDetector
    gate: AnomalyGate
    barrier: InformationBarrier
    margin: float
    onset_lookback: int = 6
    onset_reset: int = 12

    @classmethod
    def load(cls, config: DiagnosticsConfig | None = None, with_analogues: bool = True) -> DiagnosticPipeline:
        cfg = config or default_config()
        models = cfg.resolve("models")
        sig = ScenarioSignatures.load(models / "verifier")
        onset = ScenarioSignatures.load(models / "verifier" / "onset")
        nov = cfg.raw["novelty"]
        gate = AnomalyGate(
            channel_index=sig.channel_index, channel_names=sig.channels,
            rel_threshold=float(nov["rel_threshold"]), max_quiet_channels=int(nov["max_quiet_channels"]),
            drift_channels=tuple(nov.get("drift_channels", ())),
        )
        return cls(
            generator=HypothesisGenerator.load(models / "generator"),
            verifier=HypothesisVerifier(
                sig, cfg.verifier, AnalogueFinder.load(config=cfg) if with_analogues else None,
                registry=default_registry(), onset_signatures=onset,
            ),
            novelty=NoveltyDetector.load(models / "novelty"),
            gate=gate,
            barrier=InformationBarrier(config=cfg),
            margin=float(cfg.raw["selection"]["margin"]),
            onset_lookback=int(nov.get("onset_lookback_steps", 6)),
            onset_reset=int(nov.get("onset_reset_steps", 12)),
        )

    def onset_track(self, raw: np.ndarray, start: int = 0) -> np.ndarray:
        """Причинное начало события для каждой точки записи (D-042): защёлка до возврата к исходному состоянию."""
        st = stationary_points(raw, self.gate, start, self.onset_lookback)
        return onset_track(st, start, self.onset_reset, raw, self.gate)

    @property
    def versions(self) -> dict[str, str]:
        return {"generator": self.generator.version, "verifier": self.verifier.version, "novelty": self.novelty.version}

    def run(self, window: ObservationWindow, episode: EpisodeContext | None = None, with_analogues: bool = True) -> DiagnosticReport:
        lat: dict[str, float] = {}
        t = time.perf_counter()
        hs = self.generator.generate(self.barrier.restrict(window))
        lat["generator"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        verdicts = self.verifier.verify(window, hs, with_analogues=with_analogues, episode=episode)
        lat["hypothesis_verifier"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        onset = self.verifier.onset_vector(episode)
        novelty = self.novelty.assess(self.verifier.sig.features_of(window.values), onset, self.gate, window, episode)
        lat["novelty"] = (time.perf_counter() - t) * 1000

        outcome, selected, candidates = select(verdicts, novelty, self.margin)
        return DiagnosticReport(
            window_id=window.window_id, hypotheses=hs, verdicts=tuple(verdicts), novelty=novelty, outcome=outcome,
            selected=selected, candidates=candidates, episode_memory=episode is not None,
            versions=self.versions, latency_ms={k: round(v, 3) for k, v in lat.items()},
        )

    def run_batch(
        self, windows: list[ObservationWindow], episodes: list[EpisodeContext | None], with_analogues: bool = False,
    ) -> list[DiagnosticReport]:
        """То же, что run(), но генератор применяется пакетно (для benchmark). Логика выбора та же."""
        from atomic.contracts.hypothesis import HypothesisSet, ScenarioHypothesis

        if not windows:
            return []
        X = batch_features([self.barrier.restrict(w).values for w in windows])
        proba = self.generator.predict_proba(X)
        out = []
        for w, ep, p in zip(windows, episodes, proba, strict=True):
            order = np.argsort(-p, kind="stable")[: self.generator.top_k]
            hs = HypothesisSet(
                window_id=w.window_id,
                hypotheses=tuple(
                    ScenarioHypothesis(self.generator.classes[i], float(round(p[i], 6)), r, self.generator.channels)
                    for r, i in enumerate(order, start=1)
                ),
                generator_version=self.generator.version,
            )
            verdicts = self.verifier.verify(w, hs, with_analogues=with_analogues, episode=ep)
            onset = self.verifier.onset_vector(ep)
            novelty = self.novelty.assess(self.verifier.sig.features_of(w.values), onset, self.gate, w, ep)
            outcome, selected, candidates = select(verdicts, novelty, self.margin)
            out.append(DiagnosticReport(
                window_id=w.window_id, hypotheses=hs, verdicts=tuple(verdicts), novelty=novelty, outcome=outcome,
                selected=selected, candidates=candidates, episode_memory=ep is not None, versions=self.versions,
            ))
        return out
