"""
Загрузка конфигурации диагностического контура (config/diagnostics.toml).

Путь можно переопределить переменной окружения ATOMIC_DIAG_CONFIG.
Относительные пути в конфигурации отсчитываются от корня репозитория.
"""

from __future__ import annotations

import hashlib
import json
import os
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "diagnostics.toml"


@dataclass(frozen=True)
class DiagnosticsConfig:
    raw: dict[str, Any]
    path: Path

    # --- пути ---
    def resolve(self, key: str) -> Path:
        p = Path(self.raw["paths"][key])
        return p if p.is_absolute() else PROJECT_ROOT / p

    # --- генератор ---
    @property
    def allowed_channels(self) -> tuple[str, ...]:
        return tuple(self.raw["generator"]["allowed_channels"])

    @property
    def top_k(self) -> int:
        return int(self.raw["generator"]["top_k"])

    # --- окна и разбиение ---
    @property
    def window_length(self) -> int:
        return int(self.raw["windows"]["length"])

    @property
    def window_stride(self) -> int:
        return int(self.raw["windows"]["stride"])

    @property
    def split_fractions(self) -> tuple[float, float, float]:
        s = self.raw["split"]
        return float(s["train"]), float(s["val"]), float(s["test"])

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    # --- сценарии ---
    @property
    def benchmark_scenarios(self) -> tuple[str, ...]:
        return tuple(self.raw["scenarios"]["benchmark"])

    @property
    def probe_scenarios(self) -> tuple[str, ...]:
        return tuple(self.raw["scenarios"]["probe_only"])

    @property
    def scenario_titles(self) -> dict[str, str]:
        return dict(self.raw["scenarios"]["titles"])

    @property
    def verifier(self) -> dict[str, Any]:
        return dict(self.raw["verifier"])

    @property
    def version(self) -> str:
        return str(self.raw["config_version"])

    def fingerprint(self, *sections: str) -> str:
        """Хэш конфигурации для манифестов. sections — только эти разделы (по умолчанию все)."""
        payload = {s: self.raw.get(s) for s in sections} if sections else self.raw
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]

    def data_fingerprint(self) -> str:
        """Хэш разделов, от которых зависят подготовленные данные (не пороги верификатора)."""
        return self.fingerprint("seed", "windows", "split", "scenarios", "generator")


def load_config(path: str | Path | None = None) -> DiagnosticsConfig:
    p = Path(path or os.getenv("ATOMIC_DIAG_CONFIG") or DEFAULT_CONFIG)
    with p.open("rb") as fh:
        raw = tomllib.load(fh)
    return DiagnosticsConfig(raw=raw, path=p)


@lru_cache(maxsize=1)
def default_config() -> DiagnosticsConfig:
    return load_config()
