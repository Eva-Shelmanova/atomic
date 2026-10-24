"""
Реестр каналов NPPAD (config/channels.toml).

Каждый канал: каноническое имя (как в CSV), номер в README, описание,
единица, подсистема и флаг label_proxy.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from atomic.diagnostics.config import PROJECT_ROOT

DEFAULT_CHANNELS = PROJECT_ROOT / "config" / "channels.toml"


@dataclass(frozen=True)
class ChannelSpec:
    name: str
    readme_id: int
    label: str
    unit: str
    subsystem: str
    label_proxy: bool


@dataclass(frozen=True)
class ChannelRegistry:
    schema_version: str
    time_column: str
    ignored_columns: tuple[str, ...]
    channels: tuple[ChannelSpec, ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.channels)

    def get(self, name: str) -> ChannelSpec:
        for c in self.channels:
            if c.name == name:
                return c
        raise KeyError(f"канал {name!r} отсутствует в реестре")

    def label_proxies(self) -> frozenset[str]:
        return frozenset(c.name for c in self.channels if c.label_proxy)

    def observable(self) -> tuple[str, ...]:
        """Каналы, доступные штатным приборам (без label_proxy)."""
        return tuple(c.name for c in self.channels if not c.label_proxy)

    def validate_allowed(self, allowed: tuple[str, ...]) -> None:
        """Проверка конфигурации генератора: известные каналы, без label_proxy, без повторов."""
        unknown = [a for a in allowed if a not in self.names]
        if unknown:
            raise ValueError(f"неизвестные каналы в allowed_channels: {unknown}")
        proxies = sorted(set(allowed) & self.label_proxies())
        if proxies:
            raise ValueError(f"каналы label_proxy недопустимы для генератора: {proxies}")
        if len(set(allowed)) != len(allowed):
            raise ValueError("в allowed_channels есть повторы")


def load_registry(path: str | Path | None = None) -> ChannelRegistry:
    with Path(path or DEFAULT_CHANNELS).open("rb") as fh:
        raw = tomllib.load(fh)
    specs = tuple(
        ChannelSpec(
            name=c["name"],
            readme_id=int(c["readme_id"]),
            label=c["label"],
            unit=c["unit"],
            subsystem=c["subsystem"],
            label_proxy=bool(c["label_proxy"]),
        )
        for c in raw["channel"]
    )
    return ChannelRegistry(
        schema_version=raw["schema_version"],
        time_column=raw["time_column"],
        ignored_columns=tuple(raw.get("ignored_columns", [])),
        channels=specs,
    )


@lru_cache(maxsize=1)
def default_registry() -> ChannelRegistry:
    return load_registry()
