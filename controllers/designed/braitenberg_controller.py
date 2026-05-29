from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ..base import BaseController


class BraitenbergController(BaseController):
    """Simple Braitenberg-style reactive controller.

    It uses proximity asymmetry to steer away from obstacles and odor intensity
    as a weak forward drive.

    Action mapping assumed by the current environment:
    - 0 = forward
    - 1 = left
    - 2 = right
    - 3 = reverse
    """

    family = "designed"
    controller_type = "braitenberg"

    def __init__(
        self,
        obstacle_gain: float = 1.0,
        odor_gain: float = 0.25,
        clear_front_threshold: float = 0.30,
        reverse_threshold: float = 0.90,
    ):
        self.obstacle_gain = obstacle_gain
        self.odor_gain = odor_gain
        self.clear_front_threshold = clear_front_threshold
        self.reverse_threshold = reverse_threshold

    def act(self, obs: np.ndarray, info: Optional[Dict[str, Any]] = None) -> int:
        prox = obs[:8]
        odor = float(obs[8]) if len(obs) > 8 else 0.0

        front = float(prox[0])
        left_signal = float(prox[1] + prox[2] + 0.5 * prox[3])
        right_signal = float(prox[7] + prox[6] + 0.5 * prox[5])

        if front > self.reverse_threshold:
            return 3

        turn_drive = self.obstacle_gain * (left_signal - right_signal)
        forward_drive = self.odor_gain * odor - front

        if front < self.clear_front_threshold and forward_drive >= -0.05:
            return 0

        return 2 if turn_drive > 0.0 else 1
