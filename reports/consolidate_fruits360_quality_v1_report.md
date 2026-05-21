# Consolidacion Fruits-360 Quality Dataset v1

Fecha: 2026-05-18 14:02
Status: **PASS**

## Fuente

- CSV: `data/fruits360_human_review/human_labels_template.csv`
- Imagenes: `source_path` en el CSV (data_external/fruits360_original_size/)

## Destino

- Dataset: `data/quality_fruits360_human_v1/`

## Distribucion de etiquetas humanas

| Etiqueta | Count | Uso |
|----------|-------|-----|
| GOOD     | 49   | clase `good` → train/val/test |
| BAD      | 220  | clase `bad`  → train/val/test |
| REVIEW   | 31   | excluido |
| INVALID  | 0     | excluido |
| **TOTAL usable** | **269** | |

## Splits

Ratio: 70% train / 15% val / 15% test — seed=42 — estratificado por clase

| Split | good | bad | Total | good% | bad% |
|-------|------|-----|-------|-------|------|
| train | 34 | 154 | 188 | 18.1% | 81.9% |
| val | 7 | 33 | 40 | 17.5% | 82.5% |
| test | 8 | 33 | 41 | 19.5% | 80.5% |

## Imagenes copiadas

- Copiadas: 269 / 269

## Archivos generados

- `data/quality_fruits360_human_v1/train/good/` — imagenes buenas de entrenamiento
- `data/quality_fruits360_human_v1/train/bad/`  — imagenes malas de entrenamiento
- `data/quality_fruits360_human_v1/val/good/`
- `data/quality_fruits360_human_v1/val/bad/`
- `data/quality_fruits360_human_v1/test/good/`
- `data/quality_fruits360_human_v1/test/bad/`
- `data/quality_fruits360_human_v1/metadata/quality_fruits360_human_v1_master.csv`
- `data/quality_fruits360_human_v1/metadata/split_summary.csv`
- `data/quality_fruits360_human_v1/metadata/excluded_review.csv`

## Advertencia de desbalance

Ratio good:bad = 1:4 aprox.

El entrenamiento posterior **necesitara** al menos una de:
- `class_weight='balanced'` (scikit-learn) o pesos manuales en CrossEntropyLoss
- `WeightedRandomSampler` (PyTorch) para equilibrar batches
- Augmentacion fuerte de GOOD (flips, rotaciones, color jitter, etc.)

## Confirmaciones

- **NO se entrenó ningun modelo.**
- Solo se copiaron imagenes y se generaron CSVs de metadata.
- Las 31 imagenes REVIEW estan documentadas en `excluded_review.csv`.
- Trazabilidad completa: `review_id` → `source_path` → `split/class/filename`.