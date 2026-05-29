from __future__ import annotations

from typing import Any, Dict

from .base import BaseController
from .designed.braitenberg_controller import BraitenbergController
from .designed.finite_state_controller import FiniteStateController
from .designed.heuristic_controller import HeuristicController
from .designed.random_controller import RandomController
from .trained.ann_controller import ANNController
from .trained.lstm_controller import LSTMController
from .trained.transformer_controller import TransformerController


CONTROLLER_REGISTRY = {
    "random": RandomController,
    "braitenberg": BraitenbergController,
    "heuristic": HeuristicController,
    "finite_state": FiniteStateController,
    "ann": ANNController,
    "lstm": LSTMController,
    "transformer": TransformerController,
}


def make_controller(controller_type: str, **kwargs: Dict[str, Any]) -> BaseController:
    """Construct a controller from its type name."""
    if controller_type not in CONTROLLER_REGISTRY:
        available = ", ".join(sorted(CONTROLLER_REGISTRY))
        raise ValueError(f"Unknown controller_type='{controller_type}'. Available: {available}")
    return CONTROLLER_REGISTRY[controller_type](**kwargs)
