"""
Источник телеметрии для диагностического контура (только чтение, D-030).

TelemetryReplay воспроизводит один прогон NPPAD как поток наблюдений: по моменту t
отдаёт последнее завершившееся окно и память о событии (только прошлое).
Он же разрешает ссылки на окна для правила REF_RESOLVABLE верификатора ответа.
Никаких управляющих воздействий источник не допускает.
"""

from __future__ import annotations

import numpy as np

from atomic.contracts.hypothesis import EpisodeContext, ObservationWindow
from atomic.diagnostics.nppad import NPPADDataset


class NotEnoughData(ValueError):
    """К моменту t не завершилось ни одного окна наблюдения."""


class TelemetryReplay:
    def __init__(self, dataset: NPPADDataset, run_id: str, episode_memory: bool = True):
        self.ds = dataset
        self.run_id = run_id
        self.episode_memory = episode_memory
        self._windows = list(dataset.windows(run_id))
        if not self._windows:
            raise ValueError("прогон короче одного окна")
        self._ids = {w.window_id for w in self._windows} | {dataset.episode(run_id, 0.0).onset_window_id}
        self._track = None

    @property
    def windows(self) -> list[ObservationWindow]:
        return list(self._windows)

    def window_at(self, t: float) -> ObservationWindow:
        """Последнее окно, завершившееся не позже t. Раньше конца первого окна — NotEnoughData:
        отдать первое окно значило бы показать данные из будущего относительно t."""
        done = [w for w in self._windows if w.t_end <= t]
        if not done:
            raise NotEnoughData(f"первое окно завершается в {self._windows[0].t_end:.0f} с, запрошен момент {t:.0f} с")
        return done[-1]

    def episode_for(self, window: ObservationWindow, pipeline=None) -> EpisodeContext | None:
        """Память о событии для окна. С pipeline начало события определяется причинно (D-042),
        без него — начало записи (в NPPAD неисправность вводится в 0,5 с из стационара)."""
        if not self.episode_memory:
            return None
        onset = 0
        if pipeline is not None:
            if self._track is None:
                self._time, raw = self.ds.run_raw(self.run_id)
                self._track = pipeline.onset_track(raw)
            onset = int(self._track[int(np.searchsorted(self._time, window.t_end, side="right")) - 1])
        ep = self.ds.episode(self.run_id, window.t_end, onset=onset)
        self._ids.add(ep.onset_window_id)
        return ep

    def resolves(self, ref_id: str) -> bool:
        """'<window_id>' или '<window_id>#<канал>' — окно этого прогона и существующий канал."""
        wid, _, channel = ref_id.partition("#")
        return wid in self._ids and (not channel or channel in self.ds.channels)
