# Foraging Project: Evolutionary Robotics and Neural Architectures

This project simulates a foraging robot using different control architectures, ranging from hardcoded logic to evolved neural networks. The environment involves obstacle avoidance and odor-gradient following.

## 1. Install the environment

```bash
conda create -n forage python=3.11 -y
conda activate forage
pip install gymnasium numpy pygame
python -c "import gymnasium, numpy, pygame; print('ok')"
```
Note (macOS): in some cases you may need to run conda install pip before using pip install.

## 2. Project Structure

The project has been structured into a minimal set of master files to facilitate rapid experimentation:

- foraging_env.py: Contains the world logic, physics, and configuration (ForagingEnvConfig).
- controllers.py: Centralizes all "brains" (Random, Braitenberg, MLP) and the Controller Factory.
- run.py: The master execution script for all experiments.
- scripts/: Directory containing training tools like train_evolutionary_ann.py.
- models/: Directory containing evolved weights in .json format.

## 3. Sensor Architecture

The robot utilizes a 10-dimensional observation vector:
- Inputs 0-7: Proximity sensors (Ray-casting) for obstacle detection.
- Input 8: Relative angle to food (Normalized -1.0 to 1.0).
- Input 9: Exponential odor intensity (Gradient sensing).

## 4. Running Experiments

Use the master script `run.py` to execute different controller types.

### A. Evolved MLP (Reactive Brain)

Runs the master MLP weights trained over 80 generations.
```bash
python run.py --type mlp --model models/mlp_master_80gen.json --render --episodes 5
```

### B. Leaky Braitenberg (Designed Brain)

Runs a hand-coded controller with leaky-integrator neurons for smoother navigation.
```bash
python run.py --type braitenberg --render --episodes 5
```

### C. Random Baseline (Stochastic Brain)

Runs an agent with high-entropy random actions for benchmarking.
```bash
python run.py --type random --render --sleep 0.005
```
### D. Evolved GRU (Reactive Brain with Memory)

Runs GRU trained weights.
```bash
python run.py --type gru --model models/gru_master_100gen.json --render
```

## E. Evolved Residual MLP (ANN with Memory)
```bash
python run.py --type res --model models/res_mlp_100gen.json --render
```

## F. Evolved Leaky Residual MLP (ANN with Short Memory)
```bash
python run.py --type leaky_res --model models/leaky_res_final.json --render
```

## 5. Training New Models

### MLP
To evolve new neural network weights (MLP), use the training script located in the `scripts/scripts/train_evolutionary_ann.py` directory. This script utilizes multiprocessing to leverage all CPU cores.

### Execution Example
```bash
python scripts/train_evolutionary_ann.py --output models/new_mlp_model.json --generations 80 --population-size 64 --episodes-per-individual 8 --nolfi-weight 0.3 --progress-weight 0.7 --success-bonus 10.0 --mutation-std 0.05 --mutation-prob 0.2 --seed 42
```
### GRU
To evolve new neural network weights (GRU), use the training script located in the `scripts/scripts/train_evolutionary_gru.py` directory. This script utilizes multiprocessing to leverage all CPU cores.

```bash
python scripts/train_evolutionary_gru.py --output models/gru_v1.json --generations 20 --pop-size 64
```

### Residual MLP
To evolve new neural network weights (RES), use the training script located in the `scripts/train_evolutionary_res.py` directory. This script utilizes multiprocessing to leverage all CPU cores.

```bash
python scripts/train_evolutionary_res.py --generations 80 --output models/res_mlp_v1.json
```
## Residual Leaky MLP
To evolve new neural network weights (LEAKY-RES), use the training script located in the `scripts/train_evolutionary_leaky_res.py` directory. This script utilizes multiprocessing to leverage all CPU cores.

```bash
python scripts/train_evolutionary_leaky_res.py --output models/leaky_res_final.json --generations 80 --pop-size 64 --episodes 8 --mutation-std 0.05 --seed 42
```

## Plotting Fitness across Models

```bash
python plot_comparison.py
```

## Using dynamic obstacles

```bash
python run.py --type gru --model models/gru_final.json --render
```

## Using U-shape obstacle

```bash
python run.py --type gru --model models/gru_final.json --map u_shape --render
```

## Save video
```bash
python run.py --type mlp --model models/mlp_model.json --render --map default --episodes 1 --save_video --out "test.mp4"
```

## Robot Teams
```bash
python run.py --type gru --model models/gru_final.json --render --agents 1
python run.py --type random --render --agents 2
```

## How to run it

```bash
python scripts/train_social.py --type gru --agents 2 --name social_v1 --gen 50

python scripts/train_social.py --type gru --agents 2 --load models/social_v1_gen30.json --name social_v2 --gen 100

python scripts/train_social.py --type mlp --agents 2 --name mlp_social_test

python scripts/train_social.py --type gru --load models/social_gru_gen30.json --name social_faro_maestro --gen 60

python run.py --type gru --model models/GENESIS_MEGAFONO_gen10.json --render --agents 2 --map default

```

## Run Social

```bash
python run_social.py --model models/forage_social_v1_gen10.json
```
## Version
Current stable version: **1.0.0**

Check:
```bash
python run.py --version
```
