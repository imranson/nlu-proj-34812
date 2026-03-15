"""
Hyperparameter sweep for Authorship Verification (train_av.py).

Runs train_av.py with different hyperparameter combinations and
writes a summary CSV comparing all runs.

Usage:
    python code/sweep_av.py                  # run all experiments
    python code/sweep_av.py --dry_run        # print commands without running
    python code/sweep_av.py --gpu 0          # specify GPU
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path


EXPERIMENTS = {
    # Phase 1: quick sweeps with deberta-v3-base
    "base_lr": {
        "model_name": ["microsoft/deberta-v3-base"],
        "lr": [1e-5, 2e-5, 5e-5],
        "batch_size": [16],
        "max_length": [512],
        "margin": [-0.5],
        "weight_decay": [0.01],
        "epochs": [3],
    },
    "base_margin": {
        "model_name": ["microsoft/deberta-v3-base"],
        "lr": [2e-5],
        "batch_size": [16],
        "max_length": [512],
        "margin": [-0.5, 0.0, 0.5],
        "weight_decay": [0.01],
        "epochs": [3],
    },
    "base_seqlen": {
        "model_name": ["microsoft/deberta-v3-base"],
        "lr": [2e-5],
        "batch_size": [16],
        "max_length": [256, 512],
        "margin": [-0.5],
        "weight_decay": [0.01],
        "epochs": [3],
    },
    # Phase 2: try large model with promising configs
    "large_lr": {
        "model_name": ["microsoft/deberta-v3-large"],
        "lr": [1e-5, 2e-5],
        "batch_size": [8],
        "max_length": [512],
        "margin": [-0.5],
        "weight_decay": [0.01],
        "epochs": [3],
    },
}


def build_commands(experiment, gpu="0", fp16=True):
    """Generate CLI commands for all combos in an experiment."""
    keys = list(experiment.keys())
    values = list(experiment.values())
    commands = []
    for combo in itertools.product(*values):
        config = dict(zip(keys, combo))
        cmd = [sys.executable, "code/train_av.py"]
        for k, v in config.items():
            cmd += [f"--{k}", str(v)]
        if fp16:
            cmd.append("--fp16")
        cmd += ["--device", f"cuda:{gpu}" if gpu != "cpu" else "cpu"]
        commands.append((config, cmd))
    return commands


def collect_results(output_dir="checkpoints/av_siamese"):
    """Read all *_results.json files and return sorted summary."""
    results = []
    output_path = Path(output_dir)
    if not output_path.exists():
        return results
    for f in output_path.glob("*_results.json"):
        with open(f) as fh:
            data = json.load(fh)
        config = data.get("config", {})
        results.append({
            "name": f.stem.replace("_results", ""),
            "best_f1": data.get("best_f1", 0),
            "model": config.get("model_name", ""),
            "lr": config.get("lr", ""),
            "batch_size": config.get("batch_size", ""),
            "max_length": config.get("max_length", ""),
            "margin": config.get("margin", ""),
            "weight_decay": config.get("weight_decay", ""),
            "epochs": config.get("epochs", ""),
        })
    results.sort(key=lambda x: x["best_f1"], reverse=True)
    return results


def print_summary(results):
    """Print a comparison table of all runs."""
    if not results:
        print("No results found yet.")
        return
    print("\n" + "=" * 90)
    print(f"{'Rank':<5} {'Best F1':<9} {'Model':<25} {'LR':<10} {'BS':<5} {'ML':<5} {'Margin':<8} {'WD':<6}")
    print("=" * 90)
    for i, r in enumerate(results, 1):
        model_short = r["model"].split("/")[-1] if "/" in str(r["model"]) else r["model"]
        print(
            f"{i:<5} {r['best_f1']:<9.4f} {model_short:<25} "
            f"{r['lr']:<10} {r['batch_size']:<5} {r['max_length']:<5} "
            f"{r['margin']:<8} {r['weight_decay']:<6}"
        )
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS.keys()),
                        help=f"Which experiments to run. Choices: {list(EXPERIMENTS.keys())}")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no_fp16", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running")
    parser.add_argument("--summary_only", action="store_true", help="Just print results summary")
    args = parser.parse_args()

    if args.summary_only:
        print_summary(collect_results())
        return

    fp16 = args.fp16 and not args.no_fp16

    all_commands = []
    for exp_name in args.experiments:
        if exp_name not in EXPERIMENTS:
            print(f"Unknown experiment: {exp_name}. Choices: {list(EXPERIMENTS.keys())}")
            sys.exit(1)
        cmds = build_commands(EXPERIMENTS[exp_name], gpu=args.gpu, fp16=fp16)
        all_commands.extend(cmds)

    # Deduplicate by config
    seen = set()
    unique_commands = []
    for config, cmd in all_commands:
        key = tuple(sorted(config.items()))
        if key not in seen:
            seen.add(key)
            unique_commands.append((config, cmd))

    print(f"Total runs: {len(unique_commands)}")
    for i, (config, cmd) in enumerate(unique_commands, 1):
        cmd_str = " ".join(cmd)
        print(f"\n[{i}/{len(unique_commands)}] {cmd_str}")
        if args.dry_run:
            continue
        result = subprocess.run(cmd, cwd=os.getcwd())
        if result.returncode != 0:
            print(f"  WARNING: Run {i} exited with code {result.returncode}")

    if not args.dry_run:
        print_summary(collect_results())


if __name__ == "__main__":
    main()
