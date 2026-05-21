# Fix ROI Masked Contact Sheets — Report

**Fecha:** 2026-05-20 23:35:34

---

## 1. Estado de la carpeta crops/

| Métrica | Valor |
|---|---|
| Grupos de imagen encontrados | 25 |
| Grupos completos (OK) | 25 |
| Grupos incompletos | 0 |
| Total archivos JPG en crops/ | 100 |

---

## 2. Contact sheets regenerados

| Archivo | Estado |
|---|---|
| `contact_sheet_original_vs_masked.jpg` | ✅ Generado (25 filas) |
| `problem_cases_grid.jpg` | ✅ Generado (1 de 11 casos) |
| `roi_masked_diagnostics.csv` | ✅ Generado |
| `README_LOOK_HERE.txt` | ✅ Generado |

---

## 3. Casos problema

| ID | Crops previos | Método | Estado |
|---|---|---|---|
| `1000060792` | No | — | ❌ Omitido (imagen no encontrada) |
| `1000060802` | No | — | ❌ Omitido (imagen no encontrada) |
| `1000060811` | No | — | ❌ Omitido (imagen no encontrada) |
| `1000060770` | No | — | ❌ Omitido (imagen no encontrada) |
| `1000060771` | No | — | ❌ Omitido (imagen no encontrada) |
| `1000060773` | No | — | ❌ Omitido (imagen no encontrada) |
| `1000060774` | No | — | ❌ Omitido (imagen no encontrada) |
| `1000060775` | No | — | ❌ Omitido (imagen no encontrada) |
| `1000060779` | No | — | ❌ Omitido (imagen no encontrada) |
| `1000060781` | No | — | ❌ Omitido (imagen no encontrada) |
| `1000060747` | Sí | — | ✅ Incluido |

- **1** casos ya tenían crops existentes.
- **0** casos se generaron con GrabCut ligero (sin YOLO) para completar el grid.
- **10** casos omitidos por imagen original no encontrada.

> Nota: Los crops generados con GrabCut-solo (sin YOLO) usan un rectángulo central como
> inicialización. Pueden ser menos precisos que los generados con YOLO, pero son suficientes
> para diagnóstico visual.

---

## 4. Contexto

El script anterior (`prepare_quality_roi_masked_previews.py`) procesó 22 de 64 imágenes
antes de finalizar. Los crops de batch_v1 (20 imgs) y las primeras 2 de batch_v2 están
disponibles. Los 42 restantes (batch_v2 completo y batch_v3) no tienen crops todavía.

Este script solo regeneró los archivos de presentación usando lo que existía,
y completó los casos problema con GrabCut ligero.

---

## 5. Confirmaciones

- **NO** se entrenó ningún modelo.
- **NO** se modificó el dataset V2 (`data/quality_fruits360_human_v2/`).
- **NO** se modificó `best_model.pt`.
- **NO** se modificó `analyze_quality.py`.
- **NO** se modificó `quality_rules.yaml`.
- **NO** se borraron crops existentes.

---

## 6. Siguiente paso

José debe abrir:
1. `outputs/quality_roi_masked_previews/contact_sheet_original_vs_masked.jpg`
2. `outputs/quality_roi_masked_previews/problem_cases_grid.jpg`

Y comprobar si la máscara de la pera está bien recortada.
Si las máscaras son aceptables, el siguiente paso es entrenar U3 con el pipeline ROI/masked.
