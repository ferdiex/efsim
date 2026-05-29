# Controller taxonomy

This repository organizes controllers into two top-level families:

- `designed`
- `trained`

## Designed controllers

- `random`
- `braitenberg`
- `heuristic`
- `finite_state`

## Trained controllers

- `ann`
- `lstm`
- `transformer`

## Notes

At this stage, the repository includes:

- working designed-controller implementations for the current environment
- placeholders for trained-controller families
- a small factory to instantiate controllers by name

Planned next layers:

- JSON config loader
- experiment runner
- training code
- optimizers and losses
