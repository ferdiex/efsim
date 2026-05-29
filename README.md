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

### ANN experiment status

The ANN-based obstacle-avoidance experiments are currently **frozen as an exploratory baseline**.

What was tried:
- manual ANN weight tuning in `models/ann_test.json`
- render and non-render experiment configs
- turn commitment in `controllers/trained/ann_controller.py`
- obstacle safety overrides
- explicit unstuck routines

Current outcome:
- the ANN can move forward in open space
- it still gets stuck oscillating left/right near obstacles
- it is not yet a reliable obstacle-avoidance controller

Recommended next step:
- keep this ANN path as a reference baseline
- implement a robust heuristic obstacle-avoidance controller for comparison
- revisit ANN later with training, memory, or imitation learning

Relevant files:
- `controllers/trained/ann_controller.py`
- `models/ann_test.json`
- `configs/ann_test.json`
- `configs/ann_test_render.json`

## Findings: ANN imitation learning limitations

We ran a series of sanity checks and imitation-learning experiments to evaluate whether a simple feedforward ANN controller could solve the foraging task.

### Simulator sanity checks

Before blaming the learned controller, we validated the simulator itself with a deterministic debug controller:

- `forward` moves the robot consistently.
- `turn_left` and `turn_right` rotate the robot consistently in place.
- `reverse` moves the robot away from obstacles.
- proximity sensor readings change coherently with obstacle geometry and robot orientation.
- collision behavior is consistent: when the robot pushes into an obstacle, position stops changing and frontal sensors remain high.

These checks strongly suggest that the simulator dynamics, action mapping, and sensor signals are working well enough for the task.

### Imitation-learning experiments

We collected supervised datasets from the heuristic controller and trained a feedforward ANN to imitate its action choices.

What we tried:

- dataset collection from the heuristic controller with `follow_side="left"`
- dataset collection from the heuristic controller with `follow_side="right"`
- combining both datasets to reduce left/right action imbalance
- class-weighted supervised training
- frame stacking (`history=4`) to provide short-term temporal context
- optional safety overrides during ANN execution

### Result

Despite these changes, the ANN-based controller still failed to solve the task reliably:

- success rate remained `0.000` in evaluation runs
- average episode length remained at the timeout horizon
- final distance to food did not improve enough to indicate useful closed-loop behavior

This suggests that simple feedforward behavior cloning from the current heuristic policy is **not sufficient** for this task.

### Interpretation

The main issue does **not** appear to be the simulator.

Instead, the task likely requires one or more of the following:

- stronger temporal memory than a small feedforward ANN can provide
- a better expert policy
- better state coverage during data collection
- corrective data aggregation (e.g. DAgger)
- recurrent models (LSTM/GRU) or reinforcement learning
- a hybrid controller that combines heuristic obstacle negotiation with learned goal-seeking

### Practical takeaway

For now, the heuristic controller remains the strongest reliable baseline in this repository.

The ANN imitation-learning pipeline is still useful as a reproducible experiment, but it should be treated as a negative result / exploratory baseline rather than a successful controller.

## Evolutionary dense ANN pipeline

The repository now includes a basic evolutionary-robotics pipeline for a dense ANN policy:

- script: `scripts/train_evolutionary_ann.py`
- output model: `models/ann_evolutionary.json`
- eval config: `configs/ann_evolutionary.json`

Fitness combines:

- Nolfi-style forward motion term (`linear * straight * (1 - max_proximity)`)
- progress toward food (distance reduction)
- success bonus when food is reached

Train:

```bash
python scripts/train_evolutionary_ann.py --output models/ann_evolutionary.json --seed 0
```

Evaluate with the shared runner:

```bash
python run_experiment.py --config configs/ann_evolutionary.json
```

## Next steps

- implement a stronger heuristic obstacle-avoidance baseline
- revisit ANN experiments later with training or memory
- add LSTM and transformer implementations
