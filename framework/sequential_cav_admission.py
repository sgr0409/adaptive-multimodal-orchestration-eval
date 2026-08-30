"""Anytime-valid admission of fusion policies from decisive paired outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class SequentialAdmissionStats:
    selected_path: str
    admission_time: int | None
    decisive_time: int | None
    log_e_values: dict
    threshold: float


class SequentialCAVAdmission:
    """Keep a reference until a challenger's e-process crosses its threshold.

    Each update supplies paired outcomes in {-1, 0, +1}: +1 means the
    challenger alone is correct, -1 means the reference alone is correct,
    and 0 is nondecisive.  A finite mixture of likelihood-ratio e-processes
    over benefit probabilities greater than one half yields anytime-valid
    familywise false-admission control.
    """

    def __init__(self, candidate_names: Iterable[str], reference_name="equal_weight",
                 familywise_error_rate=0.05, alternative_grid=None):
        names = tuple(candidate_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("candidate_names must be nonempty and unique")
        if reference_name in names:
            raise ValueError("reference cannot be a candidate")
        if not 0 < familywise_error_rate < 1:
            raise ValueError("familywise_error_rate must lie in (0, 1)")
        grid = np.asarray(
            alternative_grid if alternative_grid is not None
            else np.linspace(0.51, 0.99, 25), dtype=float
        )
        if grid.ndim != 1 or len(grid) == 0 or np.any(grid <= 0.5) or np.any(grid >= 1):
            raise ValueError("alternative_grid values must lie strictly in (0.5, 1)")
        self.candidate_names = names
        self.reference_name = str(reference_name)
        self.alpha = float(familywise_error_rate)
        self.grid = grid
        self.log_threshold = float(np.log(len(names) / self.alpha))
        self.reset()

    def reset(self):
        self._time = 0
        self._decisive = {name: 0 for name in self.candidate_names}
        self._log_components = {
            name: np.zeros(len(self.grid), dtype=float)
            for name in self.candidate_names
        }
        self._selected = self.reference_name
        self._admission_time = None
        self._decisive_time = None
        return self

    @staticmethod
    def _logmeanexp(values):
        maximum = float(np.max(values))
        return maximum + float(np.log(np.exp(values - maximum).mean()))

    def log_e_value(self, name):
        return self._logmeanexp(self._log_components[name])

    def update(self, outcomes: Mapping[str, int]):
        if set(outcomes) != set(self.candidate_names):
            raise ValueError("outcomes must contain every declared candidate exactly once")
        self._time += 1
        if self._selected != self.reference_name:
            return self.stats()
        for name in self.candidate_names:
            outcome = int(outcomes[name])
            if outcome not in (-1, 0, 1):
                raise ValueError("paired outcomes must lie in {-1, 0, +1}")
            if outcome == 0:
                continue
            self._decisive[name] += 1
            if outcome == 1:
                self._log_components[name] += np.log(2 * self.grid)
            else:
                self._log_components[name] += np.log(2 * (1 - self.grid))
        crossed = [name for name in self.candidate_names
                   if self.log_e_value(name) >= self.log_threshold]
        if crossed:
            self._selected = max(
                crossed,
                key=lambda name: (self.log_e_value(name), -self.candidate_names.index(name)),
            )
            self._admission_time = self._time
            self._decisive_time = self._decisive[self._selected]
        return self.stats()

    def stats(self):
        return SequentialAdmissionStats(
            selected_path=self._selected,
            admission_time=self._admission_time,
            decisive_time=self._decisive_time,
            log_e_values={name: self.log_e_value(name) for name in self.candidate_names},
            threshold=float(np.exp(self.log_threshold)),
        )
