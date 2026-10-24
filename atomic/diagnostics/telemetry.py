"""
Источник телеметрии для диагностического контура (только чтение, D-030).

TelemetryReplay воспроизводит один прогон NPPAD как поток наблюдений: по моменту t
отдаёт последнее завершившееся окно и память о событии (только прошлое).
Он же разрешает ссылки на окна для правила REF_RESOLVABLE верификатора ответа.
Никаких управляющих воздействий источник не допускает.
"""

from __future__ import annotations

from atomic.contracts.hypothesis import EpisodeContext, ObservationWindow
from atomic.diagnostics.nppad import NPPADDataset


class TelemetryReplay:
    def __init__(self, dataset: NPPADDataset, run_id: str, episode_memory: bool = True):
        self.ds = dataset
        self.run_id = run_id
        self.episode_memory = episode_memory
        self._windows = list(dataset.windows(run_id))
        if not self._windows:
            raise ValueError("прогон короче одного окна")
        self._ids = {w.window_id for w in self._windows} | {dataset.episode(run_id, 0.0).onset_window_id}

    @property
    def windows(self) -> list[ObservationWindow]:
        return list(self._windows)

    def window_at(self, t: float) -> ObservationWindow:
        """Последнее окно, завершившееся не позже t (или первое окно, если t раньше)."""
        done = [w for w in self._windows if w.t_end <= t]
        return done[-1] if done else self._windows[0]

    def episode_for(self, window: ObservationWindow) -> EpisodeContext | None:
        return self.ds.episode(self.run_id, window.t_end) if self.episode_memory else None

    def resolves(self, ref_id: str) -> bool:
        """'<window_id>' или '<window_id>#<канал>' — окно этого прогона и существующий канал."""
        wid, _, channel = ref_id.partition("#")
        return wid in self._ids and (not channel or channel in self.ds.channels)
