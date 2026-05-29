from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ..base import BaseController


class LSTMController(BaseController):
    """Placeholder recurrent controller for future LSTM experiments."""

    family = "trained"
    controller_type = "lstm"

    def __init__(self, model: Any = None):
        self.model = model
        self.hidden_state = None

    def reset(self) -> None:
        self.hidden_state = None

    def act(self, obs: np.ndarray, info: Optional[Dict[str, Any]] = None) -> int:
        raise NotImplementedError(
            "LSTMController is a placeholder. Training/inference wiring will be added later."
        )
