import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from experiments.experiment_1_ambient_dimension import run_experiment_1
from experiments.experiment_2_sample_size import run_experiment_2
from experiments.experiment_3_feature_scales import run_experiment_3
from experiments.experiment_4_informative_dimension import run_experiment_4


def main():
    parser = argparse.ArgumentParser(description="Project 6: Irrelevant Variables & Curse of Dimensionality")
    parser.add_argument("--exp", default="all", choices=["1", "2", "3", "4", "all"])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()

    t0 = time.time()
    runners = {
        "1": run_experiment_1,
        "2": run_experiment_2,
        "3": run_experiment_3,
        "4": run_experiment_4,
    }

    to_run = ["1", "2", "3", "4"] if args.exp == "all" else [args.exp]
    for exp_id in to_run:
        runners[exp_id](n_seeds=args.seeds, results_dir=args.results_dir)

    print(f"\nDone in {time.time() - t0:.1f}s. Outputs saved to '{args.results_dir}/'.")


if __name__ == "__main__":
    main()
