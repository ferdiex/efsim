from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np


class BaseController(ABC):
    """Common interface for all controllers.

    The interface is intentionally small and stable so that students can swap
    controllers without changing the environment loop.
    """

    family: str = "base"
    controller_type: str = "base"

    def reset(self) -> None:
        """Reset any internal controller state at the start of an episode."""
        return None

    @abstractmethod
    def act(self, obs: np.ndarray, info: Optional[Dict[str, Any]] = None) -> int:
        """Return one discrete action for the current observation."""
        raise NotImplementedError
