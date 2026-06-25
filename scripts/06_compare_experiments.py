from __future__ import annotations

import argparse
from riceleaf.experiments.compare import save_comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs/")
    parser.add_argument("--out", default="outputs/tables/final_results.csv")
    args = parser.parse_args()
    df = save_comparison(args.runs, args.out)
    print(df)
    print(f"Saved comparison at {args.out}")


if __name__ == "__main__":
    main()
