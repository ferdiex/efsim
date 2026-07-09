# -*- coding: utf-8 -*-
import argparse
import csv
import os
from statistics import mean

def parse_pipe_list(s):
    if s is None or s == "":
        return []
    return [float(x) for x in str(s).split("|") if x != ""]

def load_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def safe_mean(values, default=float("nan")):
    return mean(values) if len(values) > 0 else default

def summarize_file(path):
    rows = load_csv(path)
    if not rows:
        return {
            "file": path,
            "episodes": 0,
            "success_rate": float("nan"),
            "avg_step_count": float("nan"),
            "avg_goal_gap": float("nan"),
            "avg_total_signals": float("nan"),
            "avg_stuck_steps": float("nan"),
            "avg_first_goal_step": float("nan"),
            "avg_second_goal_step": float("nan"),
            "social_mode": "unknown",
            "social_variant": "unknown",
            "random_spawns": "unknown",
            "u_env": "unknown",
        }

    success = [int(r["success"]) for r in rows]
    step_count = [int(r["step_count"]) for r in rows]
    stuck_steps = [int(r["stuck_steps"]) for r in rows]

    goal_gap = [int(r["goal_gap"]) for r in rows if int(r["goal_gap"]) >= 0]
    first_goal = [int(r["first_goal_step"]) for r in rows if int(r["first_goal_step"]) >= 0]
    second_goal = [int(r["second_goal_step"]) for r in rows if int(r["second_goal_step"]) >= 0]

    total_signals = []
    for r in rows:
        sigs = parse_pipe_list(r["signal_counts"])
        total_signals.append(sum(sigs))

    summary = {
        "file": path,
        "episodes": len(rows),
        "success_rate": 100.0 * safe_mean(success, 0.0),
        "avg_step_count": safe_mean(step_count),
        "avg_goal_gap": safe_mean(goal_gap),
        "avg_total_signals": safe_mean(total_signals),
        "avg_stuck_steps": safe_mean(stuck_steps),
        "avg_first_goal_step": safe_mean(first_goal),
        "avg_second_goal_step": safe_mean(second_goal),
        "social_mode": rows[0].get("social_mode", "unknown"),
        "social_variant": rows[0].get("social_variant", "unknown"),
        "random_spawns": rows[0].get("random_spawns", "unknown"),
        "u_env": rows[0].get("u_env", "unknown"),
    }
    return summary

def print_summary(summary):
    print(f"\n=== {summary['file']} ===")
    print(f"Episodios:           {summary['episodes']}")
    print(f"Social mode:         {summary['social_mode']}")
    print(f"Social variant:      {summary['social_variant']}")
    print(f"Random spawns:       {summary['random_spawns']}")
    print(f"U env:               {summary['u_env']}")
    print(f"Success rate:        {summary['success_rate']:.1f}%")
    print(f"Avg step count:      {summary['avg_step_count']:.1f}")
    if summary['avg_goal_gap'] == summary['avg_goal_gap']:
        print(f"Avg goal gap:        {summary['avg_goal_gap']:.1f}")
    else:
        print(f"Avg goal gap:        N/A")
    print(f"Avg total signals:   {summary['avg_total_signals']:.1f}")
    print(f"Avg stuck steps:     {summary['avg_stuck_steps']:.1f}")
    if summary['avg_first_goal_step'] == summary['avg_first_goal_step']:
        print(f"Avg first goal step: {summary['avg_first_goal_step']:.1f}")
    else:
        print(f"Avg first goal step: N/A")
    if summary['avg_second_goal_step'] == summary['avg_second_goal_step']:
        print(f"Avg second goal step:{summary['avg_second_goal_step']:.1f}")
    else:
        print(f"Avg second goal step:N/A")

def print_table(summaries):
    print("\n=== TABLA COMPARATIVA ===")
    header = (
        f"{'file':35} "
        f"{'mode':12} "
        f"{'variant':16} "
        f"{'rand':5} "
        f"{'succ%':>8} "
        f"{'steps':>8} "
        f"{'gap':>8} "
        f"{'signals':>8} "
        f"{'stuck':>8}"
    )
    print(header)
    print("-" * len(header))

    def fmt(x):
        return "N/A" if x != x else f"{x:.1f}"

    for s in summaries:
        file_label = os.path.basename(s["file"])
        print(
            f"{file_label:35} "
            f"{str(s['social_mode']):12} "
            f"{str(s['social_variant']):16} "
            f"{str(s['random_spawns']):5} "
            f"{s['success_rate']:8.1f} "
            f"{s['avg_step_count']:8.1f} "
            f"{fmt(s['avg_goal_gap']):>8} "
            f"{s['avg_total_signals']:8.1f} "
            f"{s['avg_stuck_steps']:8.1f}"
        )

def main():
    parser = argparse.ArgumentParser(description="Analiza CSVs de corridas del simulador")
    parser.add_argument("csv_files", nargs="+", help="Lista de archivos CSV a analizar")
    parser.add_argument("--sort", type=str, default="file",
                        choices=["file", "success", "gap", "signals", "steps", "stuck"],
                        help="Orden de la tabla comparativa")
    args = parser.parse_args()

    summaries = [summarize_file(path) for path in args.csv_files]

    if args.sort == "success":
        summaries.sort(key=lambda x: x["success_rate"], reverse=True)
    elif args.sort == "gap":
        summaries.sort(key=lambda x: (-999999 if x["avg_goal_gap"] != x["avg_goal_gap"] else x["avg_goal_gap"]))
    elif args.sort == "signals":
        summaries.sort(key=lambda x: x["avg_total_signals"], reverse=True)
    elif args.sort == "steps":
        summaries.sort(key=lambda x: x["avg_step_count"])
    elif args.sort == "stuck":
        summaries.sort(key=lambda x: x["avg_stuck_steps"])
    else:
        summaries.sort(key=lambda x: x["file"])

    for s in summaries:
        print_summary(s)

    print_table(summaries)

if __name__ == "__main__":
    main()
