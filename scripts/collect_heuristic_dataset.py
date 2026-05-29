from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from config import ForagingEnvConfig
from controllers.factory import make_controller
from foraging_env import ForagingEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Collect imitation-learning data from the heuristic controller.")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to roll out.")
    parser.add_argument(
        "--output",
        type=str,
        default="data/heuristic_dataset.npz",
        help="Output dataset path.",
    )
    parser.add_argument(
        "--follow-side",
        type=str,
        default="left",
        choices=["left", "right"],
        help="Wall-following side for the heuristic controller.",
    )
    parser.add_argument(
        "--history",
        type=int,
        default=4,
        help="Number of observation frames to stack.",
    )
    return parser.parse_args()


def stack_history(history: deque[np.ndarray]) -> np.ndarray:
    return np.concatenate(list(history), axis=0).astype(np.float32)


def main():
    args = parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = ForagingEnv(ForagingEnvConfig(), render_mode=None)
    controller = make_controller("heuristic", follow_side=args.follow_side)

    obs_list = []
    action_list = []

    try:
        for episode in range(args.episodes):
            controller.reset()
            obs, info = env.reset()
            done = False

            history = deque(
                [np.asarray(obs, dtype=np.float32) for _ in range(args.history)],
                maxlen=args.history,
            )

            while not done:
                stacked_obs = stack_history(history)
                action = controller.act(obs, info)

                obs_list.append(stacked_obs)
                action_list.append(int(action))

                next_obs, reward, terminated, truncated, next_info = env.step(action)
                done = terminated or truncated

                obs = np.asarray(next_obs, dtype=np.float32)
                history.append(obs)
                info = next_info

            print(
                f"episode={episode} "
                f"success={info['success']} "
                f"steps={info['step_count']} "
                f"dist={info['distance_to_food']:.2f}"
            )
    finally:
        env.close()

    X = np.asarray(obs_list, dtype=np.float32)
    y = np.asarray(action_list, dtype=np.int64)

    np.savez_compressed(
        output_path,
        observations=X,
        actions=y,
    )

    print(f"saved dataset to: {output_path}")
    print(f"num_samples: {len(X)}")
    print(f"obs_shape: {X.shape}")
    print(f"action_shape: {y.shape}")

    unique, counts = np.unique(y, return_counts=True)
    print("action_counts:")
    for action, count in zip(unique, counts):
        print(f"  action={action}: {count}")


if __name__ == "__main__":
    main()
