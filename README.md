# Rice Leaf Hybrid CNN

Proyecto experimental para clasificar enfermedades en hojas de arroz comparando:

- **E0:** CNN base con imagen RGB original.
- **E1:** CNN + segmentación/ROI.
- **E2:** CNN + textura explícita GLCM+DWT.
- **E3:** modelo propuesto: CNN + segmentación/ROI + textura explícita GLCM+DWT.

## Metodología

Se usa CRISP-DM adaptado a investigación experimental con estudio de ablación. La lógica del experimento es:

```text
E0 = CNN
E1 = CNN + A
E2 = CNN + B
E3 = CNN + A + B
```

Donde:

- `A = Segmentación L*a*b* + Otsu + morfología`
- `B = Textura explícita GLCM + DWT`
- `CNN = MobileNetV2`

## Instalación local

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Orden de ejecución

```bash
python scripts/00_check_environment.py
python scripts/01_prepare_dataset.py --config configs/data/rice_leaf_4class.yml
python scripts/02_generate_segmented_dataset.py --config configs/experiments/E1_segmented_cnn.yml
python scripts/03_extract_features.py --config configs/experiments/E2_cnn_texture.yml
python scripts/03_extract_features.py --config configs/experiments/E3_proposed_segmented_cnn_texture.yml
python scripts/04_train_experiment.py --config configs/experiments/E0_baseline_cnn.yml
python scripts/04_train_experiment.py --config configs/experiments/E1_segmented_cnn.yml
python scripts/04_train_experiment.py --config configs/experiments/E2_cnn_texture.yml
python scripts/04_train_experiment.py --config configs/experiments/E3_proposed_segmented_cnn_texture.yml
python scripts/06_compare_experiments.py --runs runs/
python scripts/07_export_paper_assets.py --runs runs/
```

## Estructura esperada del dataset

Coloca las imágenes en `data/raw/rice_leaf_disease/` con una carpeta por clase:

```text
data/raw/rice_leaf_disease/
├── Bacterial Blight/
├── Blast/
├── Brown Spot/
└── Tungro/
```

Luego corre `01_prepare_dataset.py` para generar splits reproducibles.
