from __future__ import annotations

from typing import Any, Dict, Optional, List

import numpy as np

from ..base import BaseController


class TransformerController(BaseController):
    """Placeholder controller for future transformer-based sequence policies."""

    family = "trained"
    controller_type = "transformer"

    def __init__(self, model: Any = None, context_window: int = 16):
        self.model = model
        self.context_window = context_window
        self.history: List[np.ndarray] = []

    def reset(self) -> None:
        self.history = []

    def act(self, obs: np.ndarray, info: Optional[Dict[str, Any]] = None) -> int:
        raise NotImplementedError(
            "TransformerController is a placeholder. Training/inference wiring will be added later."
        )
