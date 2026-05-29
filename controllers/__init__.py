"""Controller package for Forage.

This package organizes controllers into two top-level families:
- designed
- trained

The goal is to keep the taxonomy explicit and didactic.
"""

from .factory import make_controller

__all__ = ["make_controller"]
