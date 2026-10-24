"""
Верификатор гипотез (этап B2, требования 6 и 7).

Независимая проверка каждой гипотезы генератора по полному набору наблюдаемых
каналов окна (а не по 6 разрешённым). Метод детерминированный и объяснимый:
сравнение признаков окна с сигнатурой сценария (signatures.py).

Для гипотезы s:
1. опорное множество O — остальные гипотезы того же top-k (контрастная проверка:
   «чем s отличается от конкурентов»); если гипотеза одна — все прочие сценарии;
2. вес признака p — средняя по o ∈ O специфичность spec[s, o, p];
   берутся top_channels признаков с весом ≥ min_weight;
3. значение признака внутри [q05; q95] сценария s — свидетельство «за» (вес w);
   вне [q01; q99] — «против» (w); между ними — «против» с весом w/2;
4. доля подтверждений = Σw(за) / Σw(всех); статус confirmed / rejected / uncertain
   по порогам конфигурации.

К гипотезе добавляются аналоги из базы ETHZ («в аналогичных случаях наблюдалось…»).
Рекомендаций действий верификатор не формирует.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from atomic.contracts.hypothesis import (
    ChannelEvidence,
    EvidenceDirection,
    HypothesisSet,
    HypothesisVerdict,
    ObservationWindow,
    VerdictStatus,
)
from atomic.diagnostics.analogues import AnalogueFinder
from atomic.diagnostics.channels import ChannelRegistry, default_registry
from atomic.diagnostics.signatures import ScenarioSignatures

FEATURE_TITLES = {"level": "уровень", "slope": "тренд"}


class HypothesisVerifier:
    def __init__(
        self,
        signatures: ScenarioSignatures,
        params: dict[str, Any],
        analogues: AnalogueFinder | None = None,
        registry: ChannelRegistry | None = None,
    ):
        self.sig = signatures
        self.min_weight = float(params["min_weight"])
        self.top_channels = int(params["top_channels"])
        self.confirm = float(params["confirm_threshold"])
        self.reject = float(params["reject_threshold"])
        self.analogues_k = int(params.get("analogues_k", 3))
        self.analogues = analogues
        self.registry = registry or default_registry()
        self._sidx = {s: i for i, s in enumerate(signatures.scenarios)}

    @property
    def version(self) -> str:
        parts = [f"hv-1.0.0", self.sig.version]
        if self.analogues:
            parts.append(self.analogues.version)
        return "|".join(parts)

    def weights(self, scenario: str, competitors: list[str]) -> np.ndarray:
        i = self._sidx[scenario]
        others = [self._sidx[c] for c in competitors if c in self._sidx and c != scenario]
        if not others:
            others = [j for j in range(len(self.sig.scenarios)) if j != i]
        return self.sig.specificity[i, others].mean(axis=0)

    def verify_one(
        self, scenario: str, features: np.ndarray, window_id: str, competitors: list[str]
    ) -> tuple[VerdictStatus, float, list[ChannelEvidence], list[ChannelEvidence], list[str]]:
        if scenario not in self._sidx:
            return VerdictStatus.UNCERTAIN, 0.0, [], [], [f"для сценария {scenario} нет сигнатуры"]
        i = self._sidx[scenario]
        w = self.weights(scenario, competitors)
        order = [p for p in np.argsort(-w, kind="stable")[: self.top_channels] if w[p] >= self.min_weight]
        pairs = self.sig.pairs
        sup, con = [], []
        for p in order:
            q01, q05, q50, q95, q99 = self.sig.quantiles[i, p]
            x = float(features[p])
            channel, feat = pairs[p]
            if q05 <= x <= q95:
                direction, weight = EvidenceDirection.SUPPORTS, float(w[p])
                verdict = "в типичном диапазоне"
            elif q01 <= x <= q99:
                direction, weight = EvidenceDirection.CONTRADICTS, float(w[p]) / 2
                verdict = "на краю допустимого диапазона"
            else:
                direction, weight = EvidenceDirection.CONTRADICTS, float(w[p])
                verdict = "вне допустимого диапазона"
            label = self.registry.get(channel).label
            ev = ChannelEvidence(
                channel=channel, window_id=window_id, direction=direction, observed=round(x, 4),
                expected=round(float(q50), 4), weight=round(weight, 4),
                detail=(f"{label}: {FEATURE_TITLES[feat]} {x:+.2f}σ — {verdict} для {scenario} "
                        f"[{q05:+.2f}; {q95:+.2f}]"),
            )
            (sup if direction is EvidenceDirection.SUPPORTS else con).append(ev)
        total = sum(e.weight for e in sup) + sum(e.weight for e in con)
        notes = []
        if total == 0:
            return VerdictStatus.UNCERTAIN, 0.0, sup, con, ["нет различающих признаков с достаточным весом"]
        ratio = sum(e.weight for e in sup) / total
        if ratio >= self.confirm:
            status = VerdictStatus.CONFIRMED
        elif ratio <= self.reject:
            status = VerdictStatus.REJECTED
        else:
            status = VerdictStatus.UNCERTAIN
        return status, ratio, sup, con, notes

    def verify(self, window: ObservationWindow, hs: HypothesisSet, with_analogues: bool = True) -> list[HypothesisVerdict]:
        if window.window_id != hs.window_id:
            raise ValueError("HypothesisSet относится к другому окну")
        features = self.sig.features_of(window.values)
        scenarios = hs.scenarios()
        out = []
        for h in hs.hypotheses:
            status, ratio, sup, con, notes = self.verify_one(h.scenario, features, window.window_id, scenarios)
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
