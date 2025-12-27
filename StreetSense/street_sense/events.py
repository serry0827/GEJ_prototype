from dataclasses import dataclass
from typing import Optional, List

@dataclass
class Event:
    label: str
    confidence: float
    t_start: float
    t_end: float

class EventSmoother:
    """
    Turns per-window predictions into stable events using:
    - smoothing (moving average on target probability)
    - hysteresis thresholds (start/end)
    - min duration
    """
    def __init__(
        self,
        target_labels=("bike", "escooter"),
        start_th=0.70,
        end_th=0.40,
        min_duration_s=0.8,
        smooth_len=4,  # number of hops to average
    ):
        self.target_labels = set(target_labels)
        self.start_th = start_th
        self.end_th = end_th
        self.min_duration_s = min_duration_s
        self.smooth_len = smooth_len

        self._p_hist: List[float] = []
        self._active: Optional[Event] = None
        self._active_peak_conf: float = 0.0
        self._active_label: Optional[str] = None

    def _smooth_p(self, p: float) -> float:
        self._p_hist.append(p)
        if len(self._p_hist) > self.smooth_len:
            self._p_hist.pop(0)
        return sum(self._p_hist) / len(self._p_hist)

    def update(self, t_mid: float, label: str, conf: float, p_target: float) -> Optional[Event]:
        """
        t_mid: time (seconds) at center of the current window
        label/conf: model outputs
        p_target: sum probability of target labels (bike+escooter)
        """
        p_s = self._smooth_p(p_target)

        if self._active is None:
            if p_s >= self.start_th:
                # Start event
                self._active_label = label if label in self.target_labels else "unknown_target"
                self._active_peak_conf = conf
                self._active = Event(label=self._active_label, confidence=conf, t_start=t_mid, t_end=t_mid)
            return None

        # If active
        self._active.t_end = t_mid
        self._active_peak_conf = max(self._active_peak_conf, conf)

        # End condition
        if p_s <= self.end_th:
            duration = self._active.t_end - self._active.t_start
            finished = self._active
            finished.confidence = self._active_peak_conf
            self._active = None
            self._active_label = None
            self._active_peak_conf = 0.0

            if duration >= self.min_duration_s:
                return finished
            return None

        return None
