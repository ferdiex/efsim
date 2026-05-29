"""Taxonomy demo for the controller structure.

This script does not run training. It only demonstrates the controller
families and the common interface.
"""

from controllers.factory import make_controller


CONTROLLERS = [
    ("random", {}),
    ("braitenberg", {}),
    ("heuristic", {"follow_side": "left"}),
    ("finite_state", {"follow_side": "left"}),
    ("ann", {}),
    ("lstm", {}),
    ("transformer", {}),
]


def main() -> None:
    for controller_type, kwargs in CONTROLLERS:
        controller = make_controller(controller_type, **kwargs)
        print(
            f"family={controller.family:>8} "
            f"type={controller.controller_type:>12} "
            f"class={controller.__class__.__name__}"
        )


if __name__ == "__main__":
    main()
