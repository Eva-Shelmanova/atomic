"""
Информационный барьер (требование 5 задания).

Генератор гипотез получает только RestrictedWindow: значения разрешённых
каналов (из конфигурации), непрозрачный window_id и больше ничего — без run_id,
времени от начала прогона, исходного пути и меток сценария.
"""

from __future__ import annotations

from atomic.contracts.hypothesis import ObservationWindow, RestrictedWindow
from atomic.diagnostics.channels import ChannelRegistry, default_registry
from atomic.diagnostics.config import DiagnosticsConfig, default_config


class InformationBarrier:
    def __init__(self, config: DiagnosticsConfig | None = None, registry: ChannelRegistry | None = None):
        cfg = config or default_config()
        reg = registry or default_registry()
        reg.validate_allowed(cfg.allowed_channels)
        self.allowed: tuple[str, ...] = cfg.allowed_channels

    def restrict(self, window: ObservationWindow) -> RestrictedWindow:
        idx = [window.channels.index(c) for c in self.allowed]
        return RestrictedWindow(
            window_id=window.window_id,
            channels=self.allowed,
            values=window.values[:, idx].copy(),
        )


def restrict(window: ObservationWindow, barrier: InformationBarrier | None = None) -> RestrictedWindow:
    return (barrier or InformationBarrier()).restrict(window)
