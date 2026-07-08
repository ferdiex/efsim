from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from config import ForagingEnvConfig
from foraging_env import ForagingEnv


ACTION_WHEELS = {
    0: (1.0, 1.0),    # forward
    1: (-0.5, 0.5),   # turn_left
    2: (0.5, -0.5),   # turn_right
    3: (-0.5, -0.5),  # reverse
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a dense ANN controller with a simple evolutionary algorithm.")
    parser.add_argument("--output", type=str, default="models/ann_evolutionary.json", help="Output ANN model path.")
    parser.add_argument("--hidden-size", type=int, default=16, help="Hidden layer width.")
    parser.add_argument("--population-size", type=int, default=64, help="Population size.")
    parser.add_argument("--generations", type=int, default=60, help="Number of generations.")
    parser.add_argument("--episodes-per-individual", type=int, default=3, help="Episodes per fitness evaluation.")
    parser.add_argument("--elite-fraction", type=float, default=0.15, help="Elite fraction kept each generation.")
    parser.add_argument("--mutation-std", type=float, default=0.08, help="Gaussian mutation std.")
    parser.add_argument("--mutation-prob", type=float, default=0.2, help="Per-parameter mutation probability.")
    parser.add_argument("--init-scale", type=float, default=0.2, help="Initial population scale.")
    parser.add_argument("--nolfi-weight", type=float, default=0.7, help="Weight for Nolfi-style term.")
    parser.add_argument("--progress-weight", type=float, default=0.3, help="Weight for food-progress term.")
    parser.add_argument("--success-bonus", type=float, default=2.0, help="Episode bonus when food is reached.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    return parser.parse_args()


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def unpack_genome(genome: np.ndarray, layer_sizes: list[int]) -> Tuple[list[np.ndarray], list[np.ndarray]]:
    weights: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    cursor = 0

    for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
        w_size = in_dim * out_dim
        b_size = out_dim

        w = genome[cursor:cursor + w_size].reshape(in_dim, out_dim)
        cursor += w_size
        b = genome[cursor:cursor + b_size]
        cursor += b_size

        weights.append(w.astype(np.float32))
        biases.append(b.astype(np.float32))

    return weights, biases


def policy_action(obs: np.ndarray, weights: list[np.ndarray], biases: list[np.ndarray]) -> int:
    x = np.asarray(obs, dtype=np.float32)
    for i, (weight, bias) in enumerate(zip(weights, biases)):
        x = x @ weight + bias
        if i < len(weights) - 1:
            x = relu(x)
    return int(np.argmax(x))


def nolfi_step_score(action: int, obs: np.ndarray) -> float:
    v_l, v_r = ACTION_WHEELS[action]
    linear = max(0.0, (v_l + v_r) / 2.0)
    delta_v = min(1.0, abs(v_l - v_r))
    straight = 1.0 - np.sqrt(delta_v)
    obstacle_term = 1.0 - float(np.max(obs[:8]))
    
    # ADDED: Penaliza giros puros (cuando linear ≈ 0 pero delta_v alto)
    if linear < 0.05 and delta_v > 0.5:
        return -0.1 * obstacle_term
    
    return linear * straight * obstacle_term


def evaluate_genome(
    env: ForagingEnv,
    genome: np.ndarray,
    layer_sizes: list[int],
    episodes: int,
    base_seed: int,
    nolfi_weight: float,
    progress_weight: float,
    success_bonus: float,
) -> tuple[float, float]:
    weights, biases = unpack_genome(genome, layer_sizes)
    episode_scores = []
    successes = 0

    for episode_idx in range(episodes):
        obs, info = env.reset(seed=base_seed + episode_idx)
        prev_distance = float(info["distance_to_food"])
        done = False
        total_score = 0.0

        while not done:
            action = policy_action(obs, weights, biases)
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            total_score += reward # ADDED
            done = terminated or truncated

            nolfi_term = nolfi_step_score(action, obs)
            new_distance = float(next_info["distance_to_food"])
            
            # Normalized distance in relation to odour
            proximity_reward = max(0.0, 1.0 - new_distance / env.config.odor_radius) # ADDED
            # add gradient to total reward
            total_score += 0.7 * proximity_reward # ADDED
            
            # extra reward for oudor
            odor_signal = obs[-1]
            total_score += 0.7 * odor_signal #ADDDED
            
            # ADDED: penalización por estar pegado a obstáculos
            collision_penalty = -0.5 * max(obs[:-1])   # penaliza el sensor más activado
            total_score += collision_penalty

            # ADDED: bonus por espacio libre
            free_space_bonus = 0.5 * (1.0 - max(obs[:-1]))
            total_score += free_space_bonus            
            
            # ADDED: bonus for entering food zone
            if new_distance < env.config.success_threshold:
                total_score += success_bonus * 0.5  # half reward for entering the food zone
            
            progress = (prev_distance - new_distance) / max(env.config.min_spawn_distance, 1.0)
            progress = float(np.clip(progress, -1.0, 1.0))
            total_score += nolfi_weight * nolfi_term + progress_weight * progress

            prev_distance = new_distance
            obs = next_obs

            if terminated:
                successes += 1
                total_score += success_bonus

        episode_scores.append(total_score)

    return float(np.mean(episode_scores)), float(successes) / float(episodes)


def save_model(genome: np.ndarray, layer_sizes: list[int], output_path: Path) -> None:
    weights, biases = unpack_genome(genome, layer_sizes)
    model = {
        "weights": [w.tolist() for w in weights],
        "biases": [b.tolist() for b in biases],
        "history_length": 1,
        "base_obs_dim": layer_sizes[0],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(model, f, indent=2)
        f.write("\n")


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    env = ForagingEnv(ForagingEnvConfig(), render_mode=None)

    input_dim = int(env.observation_space.shape[0])
    output_dim = int(env.action_space.n)
    layer_sizes = [input_dim, args.hidden_size, output_dim]

    genome_size = 0
    for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
        genome_size += in_dim * out_dim + out_dim

    population = rng.normal(0.0, args.init_scale, size=(args.population_size, genome_size)).astype(np.float32)
    elite_count = max(1, int(args.population_size * args.elite_fraction))

    best_genome = None
    best_fitness = -np.inf

    try:
        for generation in range(args.generations):
            fitness = np.zeros(args.population_size, dtype=np.float32)
            success = np.zeros(args.population_size, dtype=np.float32)

            for i in range(args.population_size):
                f, s = evaluate_genome(
                    env=env,
                    genome=population[i],
                    layer_sizes=layer_sizes,
                    episodes=args.episodes_per_individual,
                    base_seed=args.seed + generation * 10000 + i * 100,
                    nolfi_weight=args.nolfi_weight,
                    progress_weight=args.progress_weight,
                    success_bonus=args.success_bonus,
                )
                fitness[i] = f
                success[i] = s

            order = np.argsort(fitness)[::-1]
            population = population[order]
            fitness = fitness[order]
            success = success[order]

            if float(fitness[0]) > best_fitness:
                best_fitness = float(fitness[0])
                best_genome = population[0].copy()

            print(
                f"gen={generation:03d} "
                f"best={fitness[0]:.4f} "
                f"mean={float(np.mean(fitness)):.4f} "
                f"best_success={success[0]:.3f}"
            )

            elites = population[:elite_count]
            next_population = [elites[0].copy()]

            while len(next_population) < args.population_size:
                parent = elites[rng.integers(0, elite_count)].copy()
                mutation_mask = rng.random(genome_size) < args.mutation_prob
                noise = rng.normal(0.0, args.mutation_std, size=genome_size).astype(np.float32)
                child = parent + noise * mutation_mask.astype(np.float32)
                next_population.append(child)

            population = np.stack(next_population, axis=0)
    finally:
        env.close()

    if best_genome is None:
        raise RuntimeError("Evolution did not produce a valid genome.")

    output_path = Path(args.output)
    save_model(best_genome, layer_sizes, output_path)
    print(f"saved evolved model to: {output_path}")
    print(f"best_fitness: {best_fitness:.4f}")


if __name__ == "__main__":
    main()
