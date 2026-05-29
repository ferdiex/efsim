# Forage

A small Python repository for experimenting with foraging controllers in a 2D environment.

## Current controller taxonomy

Controllers are organized into two top-level families:

- `designed`
- `trained`

### Designed controllers
- `random`
- `braitenberg`
- `heuristic`
- `finite_state`

### Trained controllers
- `ann`
- `lstm`
- `transformer`

## Current baseline status

- `heuristic` is currently the strongest manual baseline
- `finite_state` is an intermediate stateful baseline
- `braitenberg` is a reactive illustrative baseline
- `random` is the trivial baseline

## JSON experiment runner

Experiments can now be configured with JSON files and run through a shared runner.

Examples:

```bash
python run_experiment.py --config configs/random_baseline.json
python run_experiment.py --config configs/braitenberg_baseline.json
python run_experiment.py --config configs/heuristic_baseline.json
python run_experiment.py --config configs/finite_state_baseline.json
```

## Minimal ANN controller

The repository now includes a minimal feedforward ANN controller that can:

- load weights and biases from JSON
- run feedforward inference
- choose a discrete action from logits

You can generate new ANN model files with:

```bash
python scripts/init_ann_model.py --output models/ann_test.json --layers 9 8 4 --scale 0.1 --seed 0
```

## Next steps

- improve ANN experiments
- add weight generation or training utilities
- add LSTM and transformer implementations
