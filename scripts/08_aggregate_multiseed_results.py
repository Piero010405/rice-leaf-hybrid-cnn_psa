from pathlib import Path
import json
import re

import pandas as pd


RUNS_DIR = Path("runs")
OUT_DIR = Path("outputs/tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRIC_KEYS = [
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "inference_ms_per_image",
    "num_parameters",
    "model_size_mb",
    "best_epoch",
]


def infer_seed(run_dir: Path):
    m = re.search(r"_seed(\d+)", run_dir.name)
    if m:
        return int(m.group(1))
    return None


def infer_experiment(run_dir: Path):
    exp_folder = run_dir.parent.name

    mapping = {
        "E0_baseline_cnn": ("E0", "baseline_cnn"),
        "E1_segmented_cnn": ("E1", "segmented_cnn"),
        "E2_cnn_texture": ("E2", "cnn_texture"),
        "E3_proposed_segmented_cnn_texture": ("E3", "proposed_segmented_cnn_texture"),
    }

    if exp_folder in mapping:
        return mapping[exp_folder]

    parts = exp_folder.split("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]

    return exp_folder, exp_folder


def main():
    rows = []

    metric_files = sorted(RUNS_DIR.glob("**/metrics.json"))

    if not metric_files:
        raise FileNotFoundError("No se encontraron metrics.json dentro de runs/")

    for metric_file in metric_files:
        run_dir = metric_file.parent

        with open(metric_file, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        experiment_id, experiment_name = infer_experiment(run_dir)
        seed = infer_seed(run_dir)

        row = {
            "experiment_id": experiment_id,
            "experiment_name": experiment_name,
            "seed": seed,
            "run_dir": str(run_dir),
        }

        for key in METRIC_KEYS:
            row[key] = metrics.get(key)

        rows.append(row)

    df = pd.DataFrame(rows)

    df = df.sort_values(["experiment_id", "seed", "run_dir"]).reset_index(drop=True)
    df.to_csv(OUT_DIR / "multiseed_all_results.csv", index=False)

    agg_specs = {}

    for key in METRIC_KEYS:
        if key in df.columns:
            agg_specs[f"{key}_mean"] = (key, "mean")
            agg_specs[f"{key}_std"] = (key, "std")
            agg_specs[f"{key}_min"] = (key, "min")
            agg_specs[f"{key}_max"] = (key, "max")

    agg = (
        df.groupby(["experiment_id", "experiment_name"])
        .agg(
            n_runs=("seed", "count"),
            **agg_specs,
        )
        .reset_index()
        .sort_values(["macro_f1_mean", "accuracy_mean"], ascending=False)
    )

    agg.to_csv(OUT_DIR / "multiseed_summary_mean_std.csv", index=False)

    print("\n=== ALL RESULTS ===")
    print(df.to_string(index=False))

    print("\n=== SUMMARY MEAN ± STD ===")
    print(agg.to_string(index=False))

    # Paired deltas vs E0 by seed
    pivot = df.pivot_table(
        index="seed",
        columns="experiment_id",
        values="macro_f1",
        aggfunc="mean",
    )

    if "E0" in pivot.columns:
        deltas = []

        for exp in ["E1", "E2", "E3"]:
            if exp in pivot.columns:
                common = pivot[["E0", exp]].dropna()
                if len(common) > 0:
                    diff = common[exp] - common["E0"]

                    deltas.append({
                        "comparison": f"{exp} - E0",
                        "n_common_seeds": len(common),
                        "mean_delta_macro_f1": diff.mean(),
                        "std_delta_macro_f1": diff.std(),
                        "min_delta_macro_f1": diff.min(),
                        "max_delta_macro_f1": diff.max(),
                    })

        delta_df = pd.DataFrame(deltas)
        delta_df.to_csv(OUT_DIR / "paired_deltas_vs_E0.csv", index=False)

        print("\n=== PAIRED DELTAS VS E0 ===")
        print(delta_df.to_string(index=False))

    # Optional Wilcoxon if scipy exists
    try:
        from scipy.stats import wilcoxon

        tests = []

        if "E0" in pivot.columns:
            for exp in ["E1", "E2", "E3"]:
                if exp in pivot.columns:
                    common = pivot[["E0", exp]].dropna()
                    if len(common) >= 3:
                        diff = common[exp] - common["E0"]

                        if (diff == 0).all():
                            stat = None
                            p_value = 1.0
                            note = "All paired differences are zero."
                        else:
                            stat, p_value = wilcoxon(common[exp], common["E0"])
                            note = ""

                        tests.append({
                            "comparison": f"{exp} vs E0",
                            "metric": "macro_f1",
                            "n_common_seeds": len(common),
                            "wilcoxon_statistic": stat,
                            "p_value": p_value,
                            "note": note,
                        })

        tests_df = pd.DataFrame(tests)
        tests_df.to_csv(OUT_DIR / "wilcoxon_vs_E0.csv", index=False)

        print("\n=== WILCOXON TESTS VS E0 ===")
        print(tests_df.to_string(index=False))

    except Exception as e:
        print("\n[WARN] No se pudo ejecutar Wilcoxon. Motivo:", e)

    print("\nArchivos generados:")
    print(OUT_DIR / "multiseed_all_results.csv")
    print(OUT_DIR / "multiseed_summary_mean_std.csv")
    print(OUT_DIR / "paired_deltas_vs_E0.csv")
    print(OUT_DIR / "wilcoxon_vs_E0.csv")


if __name__ == "__main__":
    main()
