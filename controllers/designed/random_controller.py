from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ..base import BaseController


class RandomController(BaseController):
    """Uniformly random controller for the discrete action space."""

    family = "designed"
    controller_type = "random"

    def __init__(self, action_space_n: int = 4, seed: Optional[int] = None):
        self.action_space_n = action_space_n
        self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray, info: Optional[Dict[str, Any]] = None) -> int:
        return int(self.rng.integers(0, self.action_space_n))
