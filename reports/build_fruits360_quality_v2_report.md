# Build Fruits-360 Quality Dataset V2

Fecha: 2026-05-18 14:34

## Diferencias V1 vs V2

| ID | Cambio |
|----|--------|
| F360_0198 | BAD -> REVIEW (excluida del entrenamiento) |
| F360_0052 | GOOD -> REVIEW (excluida del entrenamiento) |
| F360_0060 | GOOD -> GOOD (sin cambio, etiqueta confirmada) |

## Distribucion V2

| Etiqueta | Count | Uso |
|----------|-------|-----|
| GOOD     | 48 | clase good -> train/val/test |
| BAD      | 219 | clase bad  -> train/val/test |
| REVIEW   | 33 | excluido |
| INVALID  | 0 | excluido |
| **TOTAL usable** | **267** | |

## Splits (seed=42, 70/15/15, estratificado)

| Split | good | bad | Total |
|-------|------|-----|-------|
| train | 34 | 153 | 187 |
| val | 7 | 33 | 40 |
| test | 7 | 33 | 40 |

## Confirmaciones
- NO se entrenó ningun modelo.
- analyze_quality.py NO fue modificado.
- quality_rules.yaml NO fue modificado.
- Imagenes copiadas: 267/267.