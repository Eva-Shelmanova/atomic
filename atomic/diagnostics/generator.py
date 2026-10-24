"""
Генератор гипотез (этап B1, требования 4 и 5).

Вход: RestrictedWindow — только разрешённые каналы, без меток (информационный барьер).
Выход: HypothesisSet — top-k сценариев «вероятна ситуация X, уверенность Y»
без рекомендаций действий (этап 2 «Soft Interpretation» документа Upgrade).

Метод: классификатор scikit-learn на признаках окна (features.py) +
температурное масштабирование вероятностей на val-выборке (калибровка против
опасности H4 «сверхуверенность»). LLM не используется (D-019).

Модуль не импортирует хранилище меток: обучение получает метки готовым
массивом от вызывающего кода (scripts/train_generator.py).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from atomic.contracts.hypothesis import HypothesisSet, RestrictedWindow, ScenarioHypothesis
from atomic.diagnostics.features import FEATURE_VERSION, feature_names, window_features

GENERATOR_SCHEMA = "1.0.0"


def make_model(kind: str, seed: int):
    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0, random_state=seed))
    if kind == "hgb":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=seed)
    raise ValueError(f"неизвестный тип модели {kind}")


def softmax_temperature(proba: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(proba, 1e-12, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


def nll(proba: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(proba[np.arange(len(y)), y], 1e-12, 1.0))))


def fit_temperature(proba: np.ndarray, y: np.ndarray) -> float:
    res = minimize_scalar(lambda t: nll(softmax_temperature(proba, t), y), bounds=(0.05, 20.0), method="bounded")
    return float(res.x)


def expected_calibration_error(proba: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    conf = proba.max(axis=1)
    correct = (proba.argmax(axis=1) == y).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def topk_accuracy(proba: np.ndarray, y: np.ndarray, k: int) -> float:
    top = np.argsort(-proba, axis=1)[:, :k]
    return float((top == y[:, None]).any(axis=1).mean())


@dataclass
class HypothesisGenerator:
    channels: tuple[str, ...]
    classes: tuple[str, ...]
    model: Any
    temperature: float = 1.0
    top_k: int = 3
    model_kind: str = "logreg"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def version(self) -> str:
        fp = self.metadata.get("fingerprint", "dev")
        return f"gen-{GENERATOR_SCHEMA}-{self.model_kind}-{fp}"

    # --- обучение (офлайн) ---
    @classmethod
    def fit(
        cls,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        channels: tuple[str, ...],
        classes: tuple[str, ...],
        kind: str,
        seed: int,
        top_k: int,
    ) -> HypothesisGenerator:
        model = make_model(kind, seed)
        model.fit(X_train, y_train)
        raw_val = cls._aligned_proba(model, X_val, len(classes))
        temperature = fit_temperature(raw_val, y_val)
        return cls(channels=channels, classes=classes, model=model, temperature=temperature, top_k=top_k, model_kind=kind)

    @staticmethod
    def _aligned_proba(model: Any, X: np.ndarray, n_classes: int) -> np.ndarray:
        """predict_proba с выравниванием по индексам классов 0..n-1."""
        p = model.predict_proba(X)
        out = np.zeros((len(X), n_classes))
        out[:, np.asarray(model.classes_, dtype=int)] = p
        return out

    def predict_proba(self, X: np.ndarray, calibrated: bool = True) -> np.ndarray:
        p = self._aligned_proba(self.model, X, len(self.classes))
        return softmax_temperature(p, self.temperature) if calibrated else p

    # --- рабочий путь ---
    def generate(self, window: RestrictedWindow) -> HypothesisSet:
        if tuple(window.channels) != self.channels:
            raise ValueError(f"окно содержит каналы {window.channels}, генератор обучен на {self.channels}")
        x = window_features(window.values)[None, :]
        proba = self.predict_proba(x)[0]
        order = np.argsort(-proba, kind="stable")[: self.top_k]
        hyps = tuple(
            ScenarioHypothesis(
                scenario=self.classes[i], confidence=float(round(proba[i], 6)), rank=r, channels_used=self.channels
            )
            for r, i in enumerate(order, start=1)
        )
        return HypothesisSet(window_id=window.window_id, hypotheses=hyps, generator_version=self.version)

    # --- сохранение ---
    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "generator.joblib"
        joblib.dump(
            {
                "channels": self.channels, "classes": self.classes, "model": self.model,
                "temperature": self.temperature, "top_k": self.top_k, "model_kind": self.model_kind,
                "metadata": self.metadata,
            },
            path,
        )
        meta = {**self.metadata, "version": self.version, "channels": list(self.channels),
                "classes": list(self.classes), "temperature": self.temperature, "top_k": self.top_k,
                "model_kind": self.model_kind, "feature_version": FEATURE_VERSION,
                "features": feature_names(self.channels)}
        (directory / "generator.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        return path

    @classmethod
    def load(cls, directory: Path) -> HypothesisGenerator:
        d = joblib.load(directory / "generator.joblib")
        return cls(**d)
