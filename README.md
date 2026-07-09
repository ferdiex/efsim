# EFSIM — Cooperative Foraging with Evolved GRU Controllers

Minimal stable repository for cooperative foraging experiments and reproducible evaluation.

## Stable Version

Current stable version: **1.0.0**

```bash
python run.py --version
```

---

## 1) Setup

```bash
conda create -n efsim python=3.11 -y
conda activate efsim
pip install -r requirements.txt
```

If you do not use `requirements.txt`, install at least:

```bash
pip install numpy pandas matplotlib seaborn pygame
```

---

## 2) Repository Structure (stable workflow)

- `foraging_env.py` — environment dynamics and task logic
- `controllers.py` — controller factory and GRU controller support
- `scripts/train_social.py` — social/cooperative training pipeline
- `run.py` — evaluation and rendering entry point
- `analyze_results.py` — CSV metrics summary
- `generate_plots.py` — figure generation
- `models/social_emergence.json` — stable GRU model
- `logs/` — experiment outputs (baseline results included)

---

## 3) Working Commands

### Quick smoke test (CSV output)
```bash
python run.py --type gru --model models/social_emergence.json --agents 2 --u_env --episodes 3 --social_mode normal --save_csv --csv_out smoke_social_emergence.csv
```

### Visual evaluation
```bash
python run.py --type gru --model models/social_emergence.json --render
```

### Baseline evaluation (20 episodes)
```bash
python run.py --type gru --model models/social_emergence.json --agents 2 --u_env --episodes 20 --social_mode normal --save_csv --csv_out logs/baseline_social_emergence_20ep.csv
```

### Analyze results
```bash
python analyze_results.py logs/baseline_social_emergence_20ep.csv
```

### Generate figures
```bash
python generate_plots.py
```

---

## 4) Training

Run cooperative training with:

```bash
python scripts/train_social.py --type gru --agents 2 --name social_emergence --gen 50
```

Example continuing from a checkpoint:

```bash
python scripts/train_social.py --type gru --agents 2 --load models/social_emergence.json --name social_emergence_v2 --gen 30
```

---

## 5) Notes

- This README documents the **stable reproducible workflow**.
- Legacy/experimental controllers and scripts are intentionally omitted from this guide.

