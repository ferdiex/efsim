from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Optional

import numpy as np

from ..base import BaseController


class ANNController(BaseController):
    """Feedforward ANN controller with optional input normalization, frame history,
    and minimal safety overrides for obstacle negotiation.
    """

    family = "trained"
    controller_type = "ann"

    def __init__(
        self,
        model: Any = None,
        model_path: Optional[str] = None,
    ):
        if model is None and model_path is None:
            raise ValueError("ANNController requires either 'model' or 'model_path'.")

        if model is None:
            model = self._load_model(model_path)

        self.weights = [np.asarray(layer, dtype=np.float32) for layer in model["weights"]]
        self.biases = [np.asarray(layer, dtype=np.float32) for layer in model["biases"]]

        if len(self.weights) != len(self.biases):
            raise ValueError("Model weights and biases must have the same number of layers.")

        self.input_mean = None
        self.input_std = None

        normalization = model.get("normalization")
        if normalization is not None:
            self.input_mean = np.asarray(normalization["mean"], dtype=np.float32)
            self.input_std = np.asarray(normalization["std"], dtype=np.float32)

        self.history_length = int(model.get("history_length", 1))
        self.base_obs_dim = int(model.get("base_obs_dim", 9))
        self.obs_history: Deque[np.ndarray] = deque(maxlen=self.history_length)

        self.last_positions: Deque[np.ndarray] = deque(maxlen=12)
        self.reverse_steps_remaining = 0
        self.turn_steps_remaining = 0
        self.turn_action = 1

    def reset(self) -> None:
        self.obs_history = deque(maxlen=self.history_length)
        self.last_positions = deque(maxlen=12)
        self.reverse_steps_remaining = 0
        self.turn_steps_remaining = 0
        self.turn_action = 1

    def act(self, obs: np.ndarray, info: Optional[Dict[str, Any]] = None) -> int:
        obs = np.asarray(obs, dtype=np.float32)

        if len(self.obs_history) == 0:
            for _ in range(self.history_length):
                self.obs_history.append(obs.copy())
        else:
            self.obs_history.append(obs.copy())

        if info is not None and info.get("robot_position") is not None:
            self.last_positions.append(np.asarray(info["robot_position"], dtype=np.float32))

        front = float(obs[0])
        front_left = float(obs[1])
        left = float(obs[2])
        front_right = float(obs[7])

        if self.reverse_steps_remaining > 0:
            self.reverse_steps_remaining -= 1
            return 3

        if self.turn_steps_remaining > 0:
            self.turn_steps_remaining -= 1
            return self.turn_action

        if self._is_stuck():
            self.reverse_steps_remaining = 3
            self.turn_steps_remaining = 6
            self.turn_action = 1 if front_right > front_left else 2
            return 3

        # Hard obstacle override
        if front > 0.72:
            self.turn_steps_remaining = 5
            self.turn_action = 1 if front_right > front_left else 2
            return self.turn_action

        # Near-obstacle override
        if front > 0.55:
            if front_left > front_right:
                return 2
            return 1

        # Wall-too-close correction
        if left > 0.80 and front < 0.40:
            return 2

        x = np.concatenate(list(self.obs_history), axis=0).astype(np.float32)

        if self.input_mean is not None and self.input_std is not None:
            x = (x - self.input_mean) / self.input_std

        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            x = x @ weight + bias
            if i < len(self.weights) - 1:
                x = np.maximum(0.0, x)

        return int(np.argmax(x))

    def _is_stuck(self) -> bool:
        if len(self.last_positions) < self.last_positions.maxlen:
            return False
        start = self.last_positions[0]
        end = self.last_positions[-1]
        dist = float(np.linalg.norm(end - start))
        return dist < 10.0

    def _load_model(self, model_path: str) -> Dict[str, Any]:
        path = Path(model_path)
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
