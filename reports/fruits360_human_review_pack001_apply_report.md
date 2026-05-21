# Fruits-360 Human Review — Aplicación PACK 001 (Fix)

Fecha: 2026-05-17 22:14
Pack aplicado: PACK 001 (IDs F360_0001 – F360_0030)
Formato IDs detectado en CSV: 4digit
Status: **PASS**

## Archivos leídos
- `data/fruits360_human_review/human_labels_template.csv`
- `data/fruits360_human_review/fruits360_human_review_master.csv`

## Archivo modificado
- `data/fruits360_human_review/human_labels_template.csv` — 30 filas actualizadas

## Nota sobre normalización de IDs
- El prompt usa IDs con 6 dígitos: `F360_000001`
- El CSV usa IDs con 4 dígitos: `F360_0001`
- El script normalizó automáticamente antes de aplicar.

## Conteos finales PACK 001

| Etiqueta | Esperado | Obtenido | IDs |
|----------|----------|----------|-----|
| GOOD | 7 | 7 OK | F360_0005, F360_0010, F360_0011, F360_0016, F360_0018, F360_0022, F360_0030 |
| BAD | 19 | 19 OK | F360_0002, F360_0004, F360_0007, F360_0008, F360_0009, F360_0012, F360_0013, F360_0015, F360_0017, F360_0019, F360_0020, F360_0021, F360_0023, F360_0024, F360_0025, F360_0026, F360_0027, F360_0028, F360_0029 |
| INVALID | 0 | 0 OK | (ninguno) |
| REVIEW | 4 | 4 OK | F360_0001, F360_0003, F360_0006, F360_0014 |

## Validaciones realizadas

- OK: 30 IDs únicos en PACK 001
- OK: Todos los IDs en rango 1–30
- OK: Sin IDs repetidos entre categorías
- OK: Todos los IDs existen en el CSV
- OK: Etiquetas pertenecen a GOOD / BAD / INVALID / REVIEW
- OK: PACK 002 intacto (GOOD=9, BAD=17, REVIEW=4)
- OK: PACK 003 intacto (GOOD=8, BAD=17, REVIEW=5)
- OK: Total etiquetado en CSV: 90/300

## Confirmaciones

- NO se entrenó ningún modelo.
- Solo se actualizó la columna `human_label` para los 30 IDs del PACK 001.
- PACK 002 y PACK 003 no fueron modificados.
- El master CSV NO fue modificado.