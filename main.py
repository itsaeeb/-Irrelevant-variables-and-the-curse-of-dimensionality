"""
Main Entrypoint for the Curse of Dimensionality & Irrelevant Variables Project.

Executes reproducible experiments:
- Experiment 1: Ambient Dimension (d in [2, 5, 10, 20, 50, 100], r = 2)
- Experiment 2: Sample Size (n in [100, 250, 500, 1000, 2000, 5000], d = 50, r = 2)
- Experiment 3: Unequal Feature Scales (sigma_u = 10, Raw vs Std vs Weighted vs Selected vs Oracle)
- Experiment 4: Informative Dimension (r in [1, 2, 4, 8], d = 50, n = 1000)

Usage:
    python main.py --all --seeds 10
    python main.py --exp 1 --seeds 10
"""

import argparse
import os
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from experiments.experiment_1_ambient_dimension import run_experiment_1
from experiments.experiment_2_sample_size import run_experiment_2
from experiments.experiment_3_feature_scales import run_experiment_3
from experiments.experiment_4_informative_dimension import run_experiment_4


def main():
    parser = argparse.ArgumentParser(
        description="Irrelevant Variables and the Curse of Dimensionality Benchmark Pipeline"
    )
    parser.add_argument(
        "--exp",
        type=str,
        default="all",
        choices=["1", "2", "3", "4", "all"],
        help="Experiment to run: 1, 2, 3, 4, or all (default: all)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=10,
        help="Number of independent random seeds per configuration (default: 10)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory to store csvs, plots, and tables (default: results)",
    )

    args = parser.parse_args()

    start_time = time.time()
    print("=" * 80)
    print(" PROJECT: IRRELEVANT VARIABLES AND THE CURSE OF DIMENSIONALITY")
    print(f" Execution Mode: Experiment(s) '{args.exp}' | Seeds: {args.seeds}")
    print(f" Results Output Path: {os.path.abspath(args.results_dir)}")
    print("=" * 80)

    os.makedirs(os.path.join(args.results_dir, "csv"), exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, "tables"), exist_ok=True)

    if args.exp in ["1", "all"]:
        t0 = time.time()
        run_experiment_1(n_seeds=args.seeds, results_dir=args.results_dir)
        print(f"[Done] Experiment 1 runtime: {time.time() - t0:.2f}s\n")

    if args.exp in ["2", "all"]:
        t0 = time.time()
        run_experiment_2(n_seeds=args.seeds, results_dir=args.results_dir)
        print(f"[Done] Experiment 2 runtime: {time.time() - t0:.2f}s\n")

    if args.exp in ["3", "all"]:
        t0 = time.time()
        run_experiment_3(n_seeds=args.seeds, results_dir=args.results_dir)
        print(f"[Done] Experiment 3 runtime: {time.time() - t0:.2f}s\n")

    if args.exp in ["4", "all"]:
        t0 = time.time()
        run_experiment_4(n_seeds=args.seeds, results_dir=args.results_dir)
        print(f"[Done] Experiment 4 runtime: {time.time() - t0:.2f}s\n")

    total_elapsed = time.time() - start_time
    print("=" * 80)
    print(f"ALL EXPERIMENTS SUCCESSFULLY EXECUTED in {total_elapsed:.2f}s!")
    print("Results and plots saved to:")
    print(f"  CSVs:   {os.path.abspath(os.path.join(args.results_dir, 'csv'))}")
    print(f"  Plots:  {os.path.abspath(os.path.join(args.results_dir, 'plots'))}")
    print(f"  Tables: {os.path.abspath(os.path.join(args.results_dir, 'tables'))}")
    print("=" * 80)


if __name__ == "__main__":
    main()

