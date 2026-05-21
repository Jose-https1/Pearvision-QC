# Auditoría Previa — Integración U3 BAD→RECHAZA Policy

**Fecha:** 2026-05-21

---

## Dónde se toma la decisión en analyze_quality.py

La decisión final se toma en `_process_one()` a través de un bloque de fusión U3.
El bloque relevante está alrededor de la línea 631 (antes del edit).

## U3 ya integrado

Sí. El clasificador U3 (MobileNetV3-small) ya estaba integrado con flag `--use-quality-u3`.
Los outputs `quality_u3_p_good`, `quality_u3_p_bad`, `quality_u3_decision_raw` ya se generaban.

## Uso anterior de u3_pred / p_good / p_bad

- `U3_BAD` en safe_mode → `u3_safe = REVIEW` (nunca RECHAZA directamente)
- `U3_BAD` + strong_defect → `u3_safe = BAD` → RECHAZA (no aplicable sin YOLO defectos)
- `U3_GOOD` + p_good >= 0.85 → PASA (ya activo)

## Dónde se insertó la nueva regla

En el bloque `if u3_raw == 'U3_BAD':`, como **nueva primera condición**:

```python
BAD_REJECT_POLICY_THR = 0.995
if p_bad >= BAD_REJECT_POLICY_THR:
    u3_safe = 'BAD'
    reason = f'U3_BAD_STRONG_REJECT(p_bad={p_bad:.3f}>={BAD_REJECT_POLICY_THR})'
elif u3_safe_mode and not strong_defect:
    u3_safe = 'REVIEW'
    ...
```

## Backup creado

`scripts/analyze_quality_backup_before_u3_bad_reject_policy_20260521_122336.py`
