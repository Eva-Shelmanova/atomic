"""
Верификатор гипотез (этап B2, требования 6 и 7; ранняя фаза — этап C, D-030).

Независимая проверка каждой гипотезы генератора по полному набору наблюдаемых
каналов (а не по 6 разрешённым). Метод детерминированный и объяснимый:
сравнение признаков с сигнатурами сценариев (signatures.py).

Два семейства свидетельств:
- по текущему окну (уровень и тренд каждого канала);
- по ранней фазе события (изменение каналов через 30/60/120 с от начала события
  относительно исходного состояния), если передан EpisodeContext.

Для гипотезы s и каждого семейства:
1. опорное множество O — остальные гипотезы того же top-k (контрастная проверка, D-023);
2. вес признака p — средняя по o ∈ O специфичность spec[s, o, p];
   берутся top_channels признаков с весом ≥ min_weight (неизвестные значения пропускаются);
3. значение внутри [q05; q95] сценария s — «за» (вес w); вне [q01; q99] — «против» (w);
   между ними — «против» (w/2);
4. доля подтверждений = Σw(за) / Σw(все) по обоим семействам; ≥ confirm — подтверждена,
   ≤ reject — отклонена, иначе — не определено.

К гипотезе добавляются аналоги из базы ETHZ. Рекомендаций действий верификатор не формирует.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from atomic.contracts.hypothesis import (
    ChannelEvidence,
    EpisodeContext,
    EvidenceDirection,
    HypothesisSet,
    HypothesisVerdict,
    ObservationWindow,
    VerdictStatus,
)
from atomic.diagnostics.analogues import AnalogueFinder
from atomic.diagnostics.channels import ChannelRegistry, default_registry
from atomic.diagnostics.signatures import ScenarioSignatures, onset_features

FEATURE_TITLES = {
    "level": "уровень",
    "slope": "тренд",
    "d30": "изменение за 30 с от начала события",
    "d60": "изменение за 60 с от начала события",
    "d120": "изменение за 120 с от начала события",
}

Result = tuple[VerdictStatus, float, list[ChannelEvidence], list[ChannelEvidence], list[str]]


class HypothesisVerifier:
    def __init__(
        self,
        signatures: ScenarioSignatures,
        params: dict[str, Any],
        analogues: AnalogueFinder | None = None,
        registry: ChannelRegistry | None = None,
        onset_signatures: ScenarioSignatures | None = None,
    ):
        self.sig = signatures
        self.onset = onset_signatures
        self.min_weight = float(params["min_weight"])
        self.top_channels = int(params["top_channels"])
        self.onset_top_channels = int(params.get("onset_top_channels", self.top_channels))
        self.confirm = float(params["confirm_threshold"])
        # без памяти о событии — более строгие параметры (D-031)
        self.fallback_min_weight = float(params.get("fallback_min_weight", self.min_weight))
        self.fallback_confirm = float(params.get("fallback_confirm_threshold", self.confirm))
        self.reject = float(params["reject_threshold"])
        self.analogues_k = int(params.get("analogues_k", 3))
        # "mean" — вес признака = средняя специфичность против всех конкурентов (D-023);
        # "pairwise" — для каждого конкурента берутся свои лучшие признаки (D-038)
        self.contrast = str(params.get("contrast", "mean"))
        self.fallback_contrast = str(params.get("fallback_contrast", "mean"))
        self.analogues = analogues
        self.registry = registry or default_registry()

    @property
    def version(self) -> str:
        parts = [f"hv-1.2.0-{self.contrast}", self.sig.version]
        if self.onset:
            parts.append("onset:" + self.onset.version)
        if self.analogues:
            parts.append(self.analogues.version)
        return "|".join(parts)

    @staticmethod
    def _weights(sig: ScenarioSignatures, scenario: str, competitors: list[str]) -> np.ndarray:
        idx = {s: i for i, s in enumerate(sig.scenarios)}
        i = idx[scenario]
        others = [idx[c] for c in competitors if c in idx and c != scenario]
        if not others:
            others = [j for j in range(len(sig.scenarios)) if j != i]
        return sig.specificity[i, others].mean(axis=0)

    def weights(self, scenario: str, competitors: list[str]) -> np.ndarray:
        return self._weights(self.sig, scenario, competitors)

    def _evidence(
        self, sig: ScenarioSignatures, scenario: str, features: np.ndarray, window_id: str,
        competitors: list[str], top_n: int, min_weight: float, contrast: str = "mean",
    ) -> tuple[list[ChannelEvidence], list[ChannelEvidence]]:
        if scenario not in sig.scenarios:
            return [], []
        i = sig.scenarios.index(scenario)
        unknown = np.isnan(features) | np.isnan(sig.quantiles[i, :, 1])
        others = [c for c in competitors if c in sig.scenarios and c != scenario]
        if contrast == "pairwise" and others:
            # для каждого конкурента — свои top_n/|O| признаков с наибольшей специфичностью против него
            per = max(1, top_n // len(others))
            chosen: dict[int, float] = {}
            for o in others:
                wo = np.where(unknown, 0.0, sig.specificity[i, sig.scenarios.index(o)])
                for p in np.argsort(-wo, kind="stable")[:per]:
                    if wo[p] >= min_weight:
                        chosen[int(p)] = max(chosen.get(int(p), 0.0), float(wo[p]))
            w = np.zeros(len(features))
            for p, val in chosen.items():
                w[p] = val
            order = sorted(chosen, key=lambda p: (-chosen[p], p))
        else:
            w = np.where(unknown, 0.0, self._weights(sig, scenario, competitors))
            order = [p for p in np.argsort(-w, kind="stable")[:top_n] if w[p] >= min_weight]
        pairs = sig.pairs
        sup, con = [], []
        for p in order:
            q01, q05, q50, q95, q99 = sig.quantiles[i, p]
            x = float(features[p])
            channel, feat = pairs[p]
            if q05 <= x <= q95:
                direction, weight, verdict = EvidenceDirection.SUPPORTS, float(w[p]), "в типичном диапазоне"
            elif q01 <= x <= q99:
                direction, weight, verdict = EvidenceDirection.CONTRADICTS, float(w[p]) / 2, "на краю допустимого диапазона"
            else:
                direction, weight, verdict = EvidenceDirection.CONTRADICTS, float(w[p]), "вне допустимого диапазона"
            label = self.registry.get(channel).label
            ev = ChannelEvidence(
                channel=channel, window_id=window_id, direction=direction, observed=float(f"{x:.4g}"),
                expected=float(f"{q50:.4g}"), weight=round(weight, 4),
                # 3 значащие цифры: изменения ранней фазы бывают порядка 10⁻³σ
                detail=(f"{label}: {FEATURE_TITLES.get(feat, feat)} {x:+.3g}σ — {verdict} для {scenario} "
                        f"[{q05:+.3g}; {q95:+.3g}]"),
            )
            (sup if direction is EvidenceDirection.SUPPORTS else con).append(ev)
        return sup, con

    def _status(self, sup: list[ChannelEvidence], con: list[ChannelEvidence], confirm: float) -> tuple[VerdictStatus, float]:
        total = sum(e.weight for e in sup) + sum(e.weight for e in con)
        if total == 0:
            return VerdictStatus.UNCERTAIN, 0.0
        ratio = sum(e.weight for e in sup) / total
        if ratio >= confirm:
            return VerdictStatus.CONFIRMED, ratio
        if ratio <= self.reject:
            return VerdictStatus.REJECTED, ratio
        return VerdictStatus.UNCERTAIN, ratio

    def verify_one(
        self, scenario: str, features: np.ndarray, window_id: str, competitors: list[str],
        onset_feats: np.ndarray | None = None, onset_window_id: str | None = None,
    ) -> Result:
        if scenario not in self.sig.scenarios:
            return VerdictStatus.UNCERTAIN, 0.0, [], [], [f"для сценария {scenario} нет сигнатуры"]
        with_memory = self.onset is not None and onset_feats is not None
        mw = self.min_weight if with_memory else self.fallback_min_weight
        confirm = self.confirm if with_memory else self.fallback_confirm
        contrast = self.contrast if with_memory else self.fallback_contrast
        sup, con = self._evidence(self.sig, scenario, features, window_id, competitors, self.top_channels, mw, contrast)
        notes: list[str] = []
        if with_memory:
            s2, c2 = self._evidence(self.onset, scenario, onset_feats, onset_window_id or window_id,
                                    competitors, self.onset_top_channels, mw, contrast)
            sup, con = sup + s2, con + c2
            if not (s2 or c2):
                notes.append("ранняя фаза события не даёт различающих признаков")
        elif self.onset is not None:
            notes.append("нет памяти о начале события: проверка только по текущему окну")
        status, ratio = self._status(sup, con, confirm)
        if not (sup or con):
            notes.append("нет различающих признаков с достаточным весом")
        return status, ratio, sup, con, notes

    def onset_vector(self, episode: EpisodeContext | None) -> np.ndarray | None:
        if episode is None or self.onset is None:
            return None
        return onset_features(episode.baseline, episode.early, self.onset.channel_index, episode.step_s)

    def verify(
        self, window: ObservationWindow, hs: HypothesisSet, with_analogues: bool = True,
        episode: EpisodeContext | None = None,
    ) -> list[HypothesisVerdict]:
        if window.window_id != hs.window_id:
            raise ValueError("HypothesisSet относится к другому окну")
        features = self.sig.features_of(window.values)
        onset_feats = self.onset_vector(episode)
        scenarios = hs.scenarios()
        out = []
        for h in hs.hypotheses:
            status, ratio, sup, con, notes = self.verify_one(
                h.scenario, features, window.window_id, scenarios, onset_feats,
                episode.onset_window_id if episode else None,
            )
            analogues = ()
            if with_analogues and self.analogues is not None:
                analogues = tuple(self.analogues.find(h.scenario, k=self.analogues_k))
            out.append(
                HypothesisVerdict(
                    hypothesis=h, status=status, support_ratio=round(ratio, 4),
                    evidence_for=tuple(sup), evidence_against=tuple(con), analogues=analogues,
                    verifier_version=self.version, notes=tuple(notes),
                )
            )
        return out
