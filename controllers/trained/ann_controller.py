from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ..base import BaseController


class ANNController(BaseController):
    """Placeholder feedforward controller.

    This class defines the slot for the future ANN controller family.
    The implementation is intentionally minimal for now.
    """

    family = "trained"
    controller_type = "ann"

    def __init__(self, model: Any = None):
        self.model = model

    def act(self, obs: np.ndarray, info: Optional[Dict[str, Any]] = None) -> int:
        raise NotImplementedError(
            "ANNController is a placeholder. Training/inference wiring will be added later."
        )
