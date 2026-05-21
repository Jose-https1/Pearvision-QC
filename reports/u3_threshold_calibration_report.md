# U3 Threshold Calibration Report

**Fecha:** 2026-05-21

## Politica de decision

```
if p_bad >= bad_reject_threshold  -> BAD
elif p_good >= good_accept_threshold -> GOOD
else                               -> REVIEW
```

## Criterio de seleccion

1. false_bad_rate == 0 (BUENA pera no va a BAD directamente)
2. bad_catch_rate >= 0.90 (al menos 90% de las malas atrapadas en BAD o REVIEW)
3. Minimizar review_rate de peras buenas
4. Maximizar bad_recall

## Umbrales seleccionados

| Parametro | Valor |
|---|---|
| bad_reject_threshold | 0.6 |
| good_accept_threshold | 0.55 |

## Resultados en validation set

| Metrica | Valor |
|---|---|
| false_bad_rate | 0.0 |
| bad_recall | 1.0 |
| bad_catch_rate | 1.0 |
| review_rate_good | 0.0 |
| bad_correct | 32 |
| bad_review | 0 |
| bad_to_good | 0 |
| good_correct | 13 |
| good_review | 0 |
| good_to_bad | 0 |

## Resultados en test set

| Metrica | Valor |
|---|---|
| false_bad_rate | 0.2 |
| bad_recall | 0.9706 |
| bad_catch_rate | 0.9706 |

## Notas

- Grid search sobre val set (no test set) para evitar data leakage.
- No se modifico V2.
