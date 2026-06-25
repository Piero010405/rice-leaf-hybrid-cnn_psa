from __future__ import annotations

import argparse
from pathlib import Path
import shutil
from riceleaf.experiments.compare import save_comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs/")
    args = parser.parse_args()
    save_comparison(args.runs, "outputs/tables/final_results.csv")
    out_cm = Path("outputs/figures/confusion_matrices")
    out_cm.mkdir(parents=True, exist_ok=True)
    for cm in Path(args.runs).rglob("confusion_matrix.png"):
        exp = cm.parent.parent.name
        shutil.copy2(cm, out_cm / f"{exp}_{cm.parent.name}.png")
    print("Paper assets exported to outputs/")


if __name__ == "__main__":
    main()
