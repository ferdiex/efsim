from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ..base import BaseController


class BraitenbergController(BaseController):
    """Simple Braitenberg-style reactive controller.

    Kept intentionally simple, but with small recovery and turn persistence so
    it does not get trapped so easily against walls.
    """

    family = "designed"
    controller_type = "braitenberg"

    def __init__(
        self,
        obstacle_gain: float = 1.0,
        odor_gain: float = 0.35,
        clear_front_threshold: float = 0.40,
        reverse_threshold: float = 0.85,
        stuck_window: int = 12,
        stuck_distance_threshold: float = 8.0,
        recovery_reverse_steps: int = 2,
        recovery_turn_steps: int = 4,
        turn_commit_steps: int = 2,
    ):
        self.obstacle_gain = obstacle_gain
        self.odor_gain = odor_gain
        self.clear_front_threshold = clear_front_threshold
        self.reverse_threshold = reverse_threshold
        self.stuck_window = stuck_window
        self.stuck_distance_threshold = stuck_distance_threshold
        self.recovery_reverse_steps = recovery_reverse_steps
        self.recovery_turn_steps = recovery_turn_steps
        self.turn_commit_steps = turn_commit_steps
        self.reset()

    def reset(self) -> None:
        self.last_positions = []
        self.reverse_steps_remaining = 0
        self.turn_steps_remaining = 0
        self.turn_action = 1

    def act(self, obs: np.ndarray, info: Optional[Dict[str, Any]] = None) -> int:
        info = info or {}
        position = info.get("robot_position")
        if position is not None:
            self._update_position_history(np.array(position, dtype=np.float32))

        if self.reverse_steps_remaining > 0:
            self.reverse_steps_remaining -= 1
            return 3

        if self.turn_steps_remaining > 0:
            self.turn_steps_remaining -= 1
            return self.turn_action

        prox = obs[:8]
        odor = float(obs[8]) if len(obs) > 8 else 0.0

        front = float(prox[0])
        front_left = float(prox[1])
        left = float(prox[2])
        back_left = float(prox[3])
        back_right = float(prox[5])
        right = float(prox[6])
        front_right = float(prox[7])

        left_signal = front_left + left + 0.5 * back_left
        right_signal = front_right + right + 0.5 * back_right

        if self._looks_stuck():
            self.reverse_steps_remaining = self.recovery_reverse_steps
            self.turn_action = 1 if left <= right else 2
            self.turn_steps_remaining = self.recovery_turn_steps
            return 3

        if front > self.reverse_threshold or max(front_left, front_right) > 0.90:
            self.reverse_steps_remaining = self.recovery_reverse_steps
            self.turn_action = 1 if left <= right else 2
            self.turn_steps_remaining = self.recovery_turn_steps
            return 3

        turn_drive = self.obstacle_gain * (left_signal - right_signal)
        forward_drive = self.odor_gain * odor - 0.7 * front

        if front < self.clear_front_threshold and abs(turn_drive) < 0.15 and forward_drive >= -0.02:
            return 0

        self.turn_action = 2 if turn_drive > 0.0 else 1
        self.turn_steps_remaining = self.turn_commit_steps
        return self.turn_action

    def _update_position_history(self, position: np.ndarray) -> None:
        self.last_positions.append(position)
        if len(self.last_positions) > self.stuck_window:
            self.last_positions.pop(0)

    def _looks_stuck(self) -> bool:
        if len(self.last_positions) < self.stuck_window:
            return False
        displacement = np.linalg.norm(self.last_positions[-1] - self.last_positions[0])
        return bool(displacement < self.stuck_distance_threshold)
    