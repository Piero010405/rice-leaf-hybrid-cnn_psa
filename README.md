# Rice Leaf Hybrid CNN

Experimental project for rice leaf disease classification using lightweight convolutional neural networks, classical segmentation, and explicit handcrafted texture descriptors.

The study compares four controlled ablation scenarios:

- **E0:** baseline CNN using the original RGB image.
- **E1:** CNN using segmented/ROI images.
- **E2:** CNN fused with explicit GLCM+DWT texture features.
- **E3:** proposed hybrid model using segmented/ROI images and explicit GLCM+DWT texture features.

## Experimental Design

The project follows a controlled ablation study:

```text
E0 = CNN
E1 = CNN + A
E2 = CNN + B
E3 = CNN + A + B
````

Where:

* `CNN = MobileNetV2`
* `A = L*a*b* segmentation + Otsu thresholding + morphology`
* `B = explicit texture descriptors: GLCM + DWT`

The objective is to evaluate the individual and combined contribution of segmentation and texture descriptors under the same dataset split, training configuration, metrics, and random seeds.

## Dataset

The experiments use the **Rice Leaf Disease Images** dataset from Kaggle:

```text
https://www.kaggle.com/datasets/nirmalsankalana/rice-leaf-disease-image
```

The dataset contains four rice disease classes:

```text
Bacterial Blight
Blast
Brown Spot
Tungro
```

The original images are not redistributed in this repository. To reproduce the experiment, download the dataset from Kaggle and place the images in the following structure:

```text
data/raw/rice_leaf_disease/
├── Bacterial Blight/
├── Blast/
├── Brown Spot/
└── Tungro/
```

## Leakage Control

A perceptual-hash-based split was used to reduce visual leakage across train, validation, and test sets. Images with the same perceptual hash were kept within the same split.

Final split distribution:

```text
Train: 4150 images
Validation: 891 images
Test: 891 images
Total: 5932 images
```

The same pHash-controlled split was used for all experiments and all seeds.

## Experimental Scenarios

| ID | Configuration                            | Purpose                 |
| -- | ---------------------------------------- | ----------------------- |
| E0 | MobileNetV2 + original RGB image         | Baseline                |
| E1 | MobileNetV2 + segmented image            | Segmentation effect     |
| E2 | MobileNetV2 + original image + GLCM+DWT  | Texture fusion effect   |
| E3 | MobileNetV2 + segmented image + GLCM+DWT | Proposed hybrid variant |

## Environment

Main environment used for the experiments:

```text
Python 3.12.1
PyTorch 2.5.1 + CUDA 12.1
GPU: NVIDIA GeForce RTX 3050 Laptop GPU
```

Install locally:

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

## Main Training Configuration

All scenarios used the same base training configuration:

```text
Image size: 224 x 224
Batch size: 8
Maximum epochs: 30
Optimizer: AdamW
Learning rate: 1e-4
Weight decay: 1e-4
Scheduler: cosine
Early stopping patience: 7
Primary metric: macro-F1
```

Each experiment was repeated using five random seeds:

```text
7, 42, 99, 123, 2026
```

This produced 20 training runs:

```text
4 experiments × 5 seeds = 20 runs
```

## Execution Order

Check the environment:

```bash
python scripts/00_check_environment.py
```

Prepare the dataset and pHash-controlled split:

```bash
python scripts/01_prepare_dataset.py --config configs/data/rice_leaf_4class.yml
```

Generate segmented images:

```bash
python scripts/02_generate_segmented_dataset.py --config configs/experiments/E1_segmented_cnn.yml
```

Extract handcrafted features:

```bash
python scripts/03_extract_features.py --config configs/experiments/E2_cnn_texture.yml
python scripts/03_extract_features.py --config configs/experiments/E3_proposed_segmented_cnn_texture.yml
```

Train individual experiments:

```bash
python scripts/04_train_experiment.py --config configs/experiments/E0_baseline_cnn.yml
python scripts/04_train_experiment.py --config configs/experiments/E1_segmented_cnn.yml
python scripts/04_train_experiment.py --config configs/experiments/E2_cnn_texture.yml
python scripts/04_train_experiment.py --config configs/experiments/E3_proposed_segmented_cnn_texture.yml
```

Compare experiments and export assets:

```bash
python scripts/06_compare_experiments.py --runs runs/
python scripts/07_export_paper_assets.py --runs runs/
```

Aggregate multi-seed results:

```bash
python scripts/08_aggregate_multiseed_results.py
```

Generate aggregate confusion matrices:

```bash
python scripts/10_aggregate_confusion_matrices.py
```

## Main Outputs

The main generated artifacts are stored in:

```text
outputs/tables/
outputs/figures/
outputs/reports/
runs/
```

Important tables include:

```text
outputs/tables/multiseed_all_results.csv
outputs/tables/multiseed_summary_mean_std.csv
outputs/tables/paired_deltas_vs_E0.csv
outputs/tables/wilcoxon_vs_E0.csv
outputs/tables/aggregate_confusion_summary.csv
```

Important figures include:

```text
outputs/figures/confusion_matrices/
outputs/figures/aggregate_confusion_matrices/
```

## Summary of Results

The main multi-seed findings were:

* **E2 CNN + Texture** achieved the highest average macro-F1.
* **E1 Segmented CNN** achieved the lowest average inference time.
* **E3 Proposed Hybrid** was competitive and improved over the baseline in aggregate accuracy, but it was not statistically superior to the baseline.
* Wilcoxon signed-rank tests did not show statistically significant improvements over E0 across five seeds.

## Repository Structure

```text
configs/        Experiment and data configuration files
data/           Local dataset directory
outputs/        Generated tables, figures, and reports
runs/           Training run outputs
scripts/        Reproducible execution scripts
src/            Source code package
paper/          LaTeX source and final paper PDF, if included
requirements.txt
README.md
```

## Notes

The original Kaggle dataset must be downloaded manually and placed under `data/raw/rice_leaf_disease/`.

Generated outputs are included to support inspection and reproducibility of the reported results.
