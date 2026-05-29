from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize a simple feedforward ANN model JSON.")
    parser.add_argument("--output", type=str, required=True, help="Path to output model JSON.")
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        required=True,
        help="Layer sizes including input and output, e.g. --layers 9 8 4",
    )
    parser.add_argument("--scale", type=float, default=0.1, help="Weight initialization scale.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if len(args.layers) < 2:
        raise ValueError("--layers must include at least input and output dimensions.")

    rng = np.random.default_rng(args.seed)
    weights = []
    biases = []

    for in_dim, out_dim in zip(args.layers[:-1], args.layers[1:]):
        weight = rng.normal(loc=0.0, scale=args.scale, size=(in_dim, out_dim)).astype(float)
        bias = np.zeros(out_dim, dtype=float)
        weights.append(weight.tolist())
        biases.append(bias.tolist())

    model = {
        "weights": weights,
        "biases": biases,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(model, f, indent=2)
        f.write("\n")

    print(f"Saved ANN model to {output_path}")
    print(f"layers={args.layers} scale={args.scale} seed={args.seed}")


if __name__ == "__main__":
    main()
