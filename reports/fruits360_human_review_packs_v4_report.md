# Fruits-360 Human Review Packs v4 — Informe

Generado: 2026-05-17 20:57

## Resumen

| Metrica | Valor |
|---------|-------|
| Total imagenes seleccionadas | 300 |
| Packs generados | 10 |
| Imagenes por pack | 30 |
| Seed de muestreo | 42 |

## Distribucion por split

| Split | Imagenes |
|-------|----------|
| Training | 149 |
| Validation | 77 |
| Test | 74 |

## Distribucion por clase Pear

| Clase | Imagenes |
|-------|----------|
| Pear 1 | 24 |
| Pear 10 | 23 |
| Pear 11 | 23 |
| Pear 12 | 23 |
| Pear 13 | 23 |
| Pear 14 | 23 |
| Pear 3 | 23 |
| Pear 5 | 23 |
| Pear 6 | 23 |
| Pear 7 | 23 |
| Pear 8 | 23 |
| Pear 9 | 23 |
| Pear common 1 | 23 |

## Packs generados

| Pack | Imagenes | Rango IDs |
|------|----------|-----------|
| 001 | 30 | F360_0001 — F360_0030 |
| 002 | 30 | F360_0031 — F360_0060 |
| 003 | 30 | F360_0061 — F360_0090 |
| 004 | 30 | F360_0091 — F360_0120 |
| 005 | 30 | F360_0121 — F360_0150 |
| 006 | 30 | F360_0151 — F360_0180 |
| 007 | 30 | F360_0181 — F360_0210 |
| 008 | 30 | F360_0211 — F360_0240 |
| 009 | 30 | F360_0241 — F360_0270 |
| 010 | 30 | F360_0271 — F360_0300 |

## Archivos generados

- `data/fruits360_human_review/images/`  (300 imagenes con ID F360_NNNN)
- `data/fruits360_human_review/fruits360_human_review_master.csv`
- `data/fruits360_human_review/human_labels_template.csv`  (rellenar este)
- `data/fruits360_human_review/README_HUMAN_LABELING.md`
- `outputs/fruits360_human_review_packs/review_pack_overview.jpg`
- `outputs/fruits360_human_review_packs/review_pack_001.jpg`
- `outputs/fruits360_human_review_packs/review_pack_002.jpg`
- `outputs/fruits360_human_review_packs/review_pack_003.jpg`
- `outputs/fruits360_human_review_packs/review_pack_004.jpg`
- `outputs/fruits360_human_review_packs/review_pack_005.jpg`
- `outputs/fruits360_human_review_packs/review_pack_006.jpg`
- `outputs/fruits360_human_review_packs/review_pack_007.jpg`
- `outputs/fruits360_human_review_packs/review_pack_008.jpg`
- `outputs/fruits360_human_review_packs/review_pack_009.jpg`
- `outputs/fruits360_human_review_packs/review_pack_010.jpg`

## Instrucciones de uso

1. Abre los contact sheets en `outputs/fruits360_human_review_packs/`.
2. Cada imagen muestra su ID `F360_NNNN` y la clase Pear de origen.
3. Rellena `human_labels_template.csv` con: GOOD / BAD / INVALID / REVIEW.
4. Una vez etiquetado, el CSV se usará para entrenar el clasificador de calidad.

## Notas

- El muestreo es equilibrado entre clases pero no ha sido revisado imagen a imagen.
- Algunas clases de Fruits-360 contienen peras con defectos reales o no comerciales.
- `INVALID` es para imágenes inutilizables (imagen negra, cortada, etc.).
- `REVIEW` es para casos dudosos que necesitan segunda opinión.