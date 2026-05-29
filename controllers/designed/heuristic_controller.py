from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ..base import BaseController


class HeuristicController(BaseController):
    """Odor-aware wall-following heuristic.

    This is the current baseline heuristic moved into the controller taxonomy.
    """

    family = "designed"
    controller_type = "heuristic"

    def __init__(self, follow_side: str = "left"):
        assert follow_side in ("left", "right")
        self.follow_side = follow_side
        self.reset()

    def reset(self) -> None:
        self.turn_steps_remaining = 0
        self.turn_action = None
        self.reverse_steps_remaining = 0
        self.last_positions = []
        self.stuck_window = 15
        self.stuck_distance_threshold = 8.0
        self.prev_odor = None
        self.best_odor = 0.0
        self.explore_turn_steps_remaining = 0
        self.explore_turn_action = None

    def act(self, obs: np.ndarray, info: Optional[Dict[str, Any]] = None) -> int:
        info = info or {}
        prox = obs[:8]
        odor = float(obs[8])

        front = prox[0]
        front_left = prox[1]
        left = prox[2]
        back_left = prox[3]
        _back = prox[4]
        back_right = prox[5]
        right = prox[6]
        front_right = prox[7]

        position = info.get("robot_position", None)
        if position is not None:
            self._update_position_history(np.array(position, dtype=np.float32))

        if odor > self.best_odor:
            self.best_odor = odor

        if self.reverse_steps_remaining > 0:
            self.reverse_steps_remaining -= 1
            self.prev_odor = odor
            return 3

        if self.turn_steps_remaining > 0:
            self.turn_steps_remaining -= 1
            self.prev_odor = odor
            return self.turn_action

        if self.explore_turn_steps_remaining > 0:
            self.explore_turn_steps_remaining -= 1
            self.prev_odor = odor
            return self.explore_turn_action

        if self._looks_stuck():
            self.reverse_steps_remaining = 3
            self.turn_action = 1 if self.follow_side == "left" else 2
            self.turn_steps_remaining = 4
            self.prev_odor = odor
            return 3

        strong_front_block = 0.55
        moderate_front_block = 0.30
        side_block = 0.75
        wall_band = 0.20

        if self.follow_side == "left":
            preferred_turn = 1
            opposite_turn = 2
            follow_sensor = left
            front_corner = front_left
        else:
            preferred_turn = 2
            opposite_turn = 1
            follow_sensor = right
            front_corner = front_right

        if front > strong_front_block or front_corner > strong_front_block:
            self.turn_action = preferred_turn
            self.turn_steps_remaining = 3
            self.prev_odor = odor
            return self.turn_action

        if follow_sensor > side_block:
            self.turn_action = opposite_turn
            self.turn_steps_remaining = 2
            self.prev_odor = odor
            return self.turn_action

        if odor > 0.0 and self.prev_odor is not None:
            odor_delta = odor - self.prev_odor

            if odor_delta > 0.01 and front < moderate_front_block:
                self.prev_odor = odor
                return 0

            if odor_delta < -0.01:
                self.explore_turn_action = preferred_turn
                self.explore_turn_steps_remaining = 2
                self.prev_odor = odor
                return self.explore_turn_action

        if 0.0 < follow_sensor < wall_band and front < moderate_front_block:
            self.turn_action = preferred_turn
            self.turn_steps_remaining = 1
            self.prev_odor = odor
            return self.turn_action

        if front < moderate_front_block:
            self.prev_odor = odor
            return 0

        self.turn_action = preferred_turn
        self.turn_steps_remaining = 2
        self.prev_odor = odor
        return self.turn_action

    def _update_position_history(self, position: np.ndarray) -> None:
        self.last_positions.append(position)
        if len(self.last_positions) > self.stuck_window:
            self.last_positions.pop(0)

    def _looks_stuck(self) -> bool:
        if len(self.last_positions) < self.stuck_window:
            return False

        start = self.last_positions[0]
        end = self.last_positions[-1]
        displacement = np.linalg.norm(end - start)
        return bool(displacement < self.stuck_distance_threshold)
