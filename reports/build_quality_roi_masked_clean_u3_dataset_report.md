# Build U3 Dataset Report

**Fecha:** 2026-05-21

## Conteos por split

| split | good | bad | total |
|---|---|---|---|
| train | 62 | 153 | 215 |
| val | 13 | 32 | 45 |
| test | 15 | 34 | 49 |
| holdout_supermarket | 22 | 0 | 22 |

## Fuentes

- batch_v3: 22 imagenes
- fruits360_v2: 267 imagenes
- hard_v1: 20 imagenes
- hard_v2: 22 imagenes

## Errores

- Total errores: 0

## Metodo de masking

- Fruits-360 (F360_*): umbral blanco (RGB > 235) + fondo gris 128
- Supermarket (1000060*): gray_bg_clean reutilizado de quality_roi_masked_previews_v2

## Notas

- No se modifico V2.
- No se modifico analyze_quality.py ni quality_rules.yaml.
- Batch_v3 excluido de train/val/test.
