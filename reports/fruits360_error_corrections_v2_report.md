# Correcciones de Errores V2 — Fruits-360

Fecha: 2026-05-18 14:33
Status: **PASS**

## Correcciones aplicadas

| review_id | Etiqueta anterior | Etiqueta nueva | Accion |
|-----------|-------------------|----------------|--------|
| F360_0198 | BAD | REVIEW | CAMBIADO |
| F360_0052 | GOOD | REVIEW | CAMBIADO |
| F360_0060 | GOOD | GOOD | SIN CAMBIO (confirmado) |

## Conteos globales post-correccion

| Etiqueta | Count |
|----------|-------|
| GOOD | 48 |
| BAD | 219 |
| REVIEW | 33 |
| INVALID | 0 |
| **TOTAL** | **300** |

## Motivacion de cada correccion

- **F360_0198** (BAD->REVIEW): clasificador V1 la predijo como GOOD con confianza 0.995.
  Inspeccion visual confirma que es un caso ambiguo — se mueve a REVIEW.
- **F360_0052** (GOOD->REVIEW): clasificador V1 la predijo como BAD con confianza 0.665.
  Inspeccion visual confirma ambiguedad — se mueve a REVIEW.
- **F360_0060** (GOOD->GOOD): clasificador V1 la predijo como BAD con confianza 0.957.
  Inspeccion visual confirma que SI es GOOD — etiqueta correcta, modelo fallaba.

## Confirmaciones

- analyze_quality.py NO fue modificado.
- quality_rules.yaml NO fue modificado.
- NO se entrenó ningun modelo.