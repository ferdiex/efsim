from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ..base import BaseController


class DebugController(BaseController):
    """Deterministic controller for simulator sanity checks."""

    family = "designed"
    controller_type = "debug"

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.plan: List[int] = (
            [0] * 20 +   # forward
            [1] * 10 +   # turn_left
            [0] * 20 +   # forward
            [2] * 10 +   # turn_right
            [0] * 20 +   # forward
            [3] * 10     # reverse
        )
        self.step_idx = 0

    def act(self, obs: np.ndarray, info: Optional[Dict[str, Any]] = None) -> int:
        action = self.plan[self.step_idx % len(self.plan)]
        self.step_idx += 1
        return action
