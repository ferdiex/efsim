from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ..base import BaseController


class FiniteStateController(BaseController):
    """Simple finite-state controller.

    States:
    - explore: move forward when clear
    - avoid: persistent turn to escape local obstacle structure
    - recover: brief reverse when stuck
    """

    family = "designed"
    controller_type = "finite_state"

    def __init__(self, follow_side: str = "left"):
        assert follow_side in ("left", "right")
        self.follow_side = follow_side
        self.reset()

    def reset(self) -> None:
        self.state = "explore"
        self.state_steps = 0
        self.last_positions = []
        self.stuck_window = 12
        self.stuck_distance_threshold = 8.0

    def act(self, obs: np.ndarray, info: Optional[Dict[str, Any]] = None) -> int:
        info = info or {}
        prox = obs[:8]
        front = float(prox[0])
        front_left = float(prox[1])
        left = float(prox[2])
        right = float(prox[6])
        front_right = float(prox[7])

        position = info.get("robot_position")
        if position is not None:
            self._update_position_history(np.array(position, dtype=np.float32))

        preferred_turn = 1 if self.follow_side == "left" else 2
        opposite_turn = 2 if self.follow_side == "left" else 1

        if self._looks_stuck() and self.state != "recover":
            self.state = "recover"
            self.state_steps = 3

        if self.state == "recover":
            self.state_steps -= 1
            if self.state_steps <= 0:
                self.state = "avoid"
                self.state_steps = 4
            return 3

        if front > 0.55 or max(front_left, front_right) > 0.60:
            self.state = "avoid"
            self.state_steps = max(self.state_steps, 3)

        if self.state == "avoid":
            self.state_steps -= 1
            if self.state_steps <= 0:
                self.state = "explore"
            side_pressure = left if self.follow_side == "left" else right
            if side_pressure > 0.80:
                return opposite_turn
            return preferred_turn

        if front < 0.30:
            return 0

        self.state = "avoid"
        self.state_steps = 2
        return preferred_turn

    def _update_position_history(self, position: np.ndarray) -> None:
        self.last_positions.append(position)
        if len(self.last_positions) > self.stuck_window:
            self.last_positions.pop(0)

    def _looks_stuck(self) -> bool:
        if len(self.last_positions) < self.stuck_window:
            return False
        displacement = np.linalg.norm(self.last_positions[-1] - self.last_positions[0])
        return bool(displacement < self.stuck_distance_threshold)
