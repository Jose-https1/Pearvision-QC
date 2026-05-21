# U3 Input Sources Audit

**Fecha:** 2026-05-21
**Propósito:** Inventario de fuentes de datos disponibles para construir el dataset U3 ROI/masked clean.

---

## 1. Fuentes encontradas

### data/quality_fruits360_human_v2/
- **Estado:** EXISTE
- **Imágenes:** 267 total — 48 GOOD, 219 BAD
- **Etiquetas:** humanas verificadas (human_label: GOOD/BAD)
- **Tipo de imagen:** Fruits-360 original_size (resolución variable, fondo blanco)
- **Archivo metadata:** `metadata/quality_fruits360_human_v2_master.csv`
- **Split actual:** train=187, val=40, test=40 (split V2, no se respetará en U3)
- **Uso para U3:** train/val/test (aplicar masking a fondo gris)
- **Nota:** Imágenes con fondo blanco limpio → umbral de blanco suficiente para masking

### data/quality_fruits360_human_u1/
- **Estado:** NO EXISTE
- **Uso:** No disponible

### data/fruits360_human_review/
- **Estado:** EXISTE (3 archivos)
- **Contenido:** fruits360_human_review_master.csv, human_labels_template.csv, README_HUMAN_LABELING.md
- **Uso para U3:** Solo referencia. No contiene imágenes directamente usables en esta iteración.

### data/supermarket_good_hard_examples_v1/images/
- **Estado:** EXISTE
- **Imágenes:** 20 imágenes GOOD (IDs 1000060736–1000060756)
- **Tipo:** Peras de supermercado reales (fondo mixto)
- **Versión masked:** disponible en `outputs/quality_roi_masked_previews_v2/crops/*_gray_bg_clean.jpg`
- **Uso para U3:** train/val/test GOOD (vía gray_bg_clean)

### data/supermarket_good_hard_examples_v2/images/
- **Estado:** EXISTE
- **Imágenes:** 22 imágenes GOOD (IDs 1000060759–1000060784 aprox.)
- **Tipo:** Peras de supermercado reales (fondo azul/negro/blanco)
- **Versión masked:** disponible en `outputs/quality_roi_masked_previews_v2/crops/*_gray_bg_clean.jpg`
- **Uso para U3:** train/val/test GOOD (vía gray_bg_clean)

### data/unseen_quality_eval_input/supermarket_valid_conditions_batch_v3/
- **Estado:** EXISTE
- **Imágenes:** 22 imágenes GOOD (IDs 1000060790–1000060815)
- **Tipo:** Peras de supermercado reales (condiciones controladas, fondo blanco)
- **Versión masked:** disponible en `outputs/quality_roi_masked_previews_v2/crops/*_gray_bg_clean.jpg`
- **Uso para U3:** SOLO holdout externo. NO entrar en train/val/test.

### outputs/quality_roi_masked_previews_v2/crops/
- **Estado:** EXISTE
- **Imágenes:** 448 archivos (64 IDs × 7 variantes)
- **gray_bg_clean:** 64 imágenes `*_gray_bg_clean.jpg` disponibles
- **Uso para U3:** fuente de gray_bg_clean para imágenes de supermercado

### outputs/quality_roi_masked_previews_v2/roi_masked_v2_diagnostics.csv
- **Estado:** EXISTE
- **Uso:** trazabilidad de máscaras; confirma OK=64, REVIEW=0, FAIL=0

---

## 2. Resumen de imágenes disponibles por uso

| Fuente | Clase | Cantidad | Uso U3 |
|---|---|---|---|
| quality_fruits360_human_v2 GOOD | good | 48 | train/val/test (tras masking) |
| quality_fruits360_human_v2 BAD | bad | 219 | train/val/test (tras masking) |
| hard_examples_v1 | good | 20 | train/val/test (gray_bg_clean) |
| hard_examples_v2 | good | 22 | train/val/test (gray_bg_clean) |
| batch_v3 (supermarket holdout) | good | 22 | SOLO holdout |
| **Total train/val/test GOOD** | good | **90** | |
| **Total train/val/test BAD** | bad | **219** | |
| **Holdout supermarket** | good | **22** | SOLO holdout |

Ratio GOOD:BAD en train/val/test ≈ 1:2.4 (manejable con class weights / sampler).

---

## 3. Advertencias

1. `data/quality_fruits360_human_u1/` NO EXISTE — se usará V2 como base principal.
2. Los 22 imágenes de batch_v3 están TAMBIÉN en los gray_bg_clean (IDs 1000060790–1000060815). El script de construcción debe excluirlas del split train/val/test.
3. Los F360 images son de resolución variable (569×432 a 776×906) — se redimensionarán a 224×224 durante el masking.
4. El masking para F360 usa umbral de blanco (todos los canales RGB > 240 → fondo) en lugar de GrabCut, porque el fondo blanco puro de Fruits-360 lo hace innecesario.
