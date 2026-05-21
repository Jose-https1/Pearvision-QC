# Auditoría de Datos — U3 BAD to REJECT Policy

**Fecha:** 2026-05-21

---

## Columnas disponibles

`image_name`, `human_label`, `image_path`, `split`, `class`, `u3_status`, `u3_pred`, `u3_p_good`, `u3_p_bad`, `u3_raw`, `strong_defect_evidence`, `final_decision`, `reason`, `business_result`, `human_label_original`, `label_corrected`, `business_result_corrected`

## Distribución de etiquetas corregidas

| | N |
|---|---|
| GOOD (corregido) | 55 |
| BAD (corregido) | 214 |
| Total | 269 |

## Distribución de final_decision (baseline)

- REVISAR: 218
- PASA: 51

## Distribución de u3_pred

- bad: 215
- good: 54

## Rangos de confianza U3

- u3_p_good: 0.0001 – 1.0000 (media 0.2120)
- u3_p_bad:  0.0000 – 0.9999 (media 0.7880)

## BAD → REVISAR con u3_pred=bad

Total: 212
- p_bad >= 0.995: 129
- p_bad >= 0.975: 174
- p_bad >= 0.950: 189
- p_bad >= 0.900: 201

## GOOD → REVISAR (casos críticos para calibración)

Total: 4

| Imagen | u3_pred | p_good | p_bad | Nota |
|---|---|---|---|---|
| F360_0018.jpg | bad | 0.1555 | 0.8445 | RISKY — bloquea umbral agresivo |
| F360_0048.jpg | bad | 0.0246 | 0.9754 | RISKY — bloquea umbral agresivo |
| F360_0060.jpg | bad | 0.0057 | 0.9943 | RISKY — bloquea umbral agresivo |
| F360_0124.jpg | good | 0.6568 | 0.3432 | SAFE — u3_pred=good |

**Nota:** Para mantener GOOD->RECHAZA=0, el umbral debe ser > 0.9943 (mayor que el p_bad máximo de estos casos).
El umbral seguro mínimo es **0.995**.
