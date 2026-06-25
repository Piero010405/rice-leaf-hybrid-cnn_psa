from __future__ import annotations

import argparse
from riceleaf.experiments.runner import run_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_dir = run_experiment(args.config)
    print(f"Run completed: {run_dir}")


if __name__ == "__main__":
    main()
