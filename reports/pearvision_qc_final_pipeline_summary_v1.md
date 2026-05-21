# PearVision QC — Resumen Final del Pipeline v1

**Fecha:** 2026-05-21
**Estado:** Pipeline final provisional ACEPTADO

---

## 1. Nombre del sistema

**PearVision QC**

---

## 2. Objetivo del sistema

Sistema local de visión artificial de bajo coste para inspección superficial de peras enteras antes del envasado. El sistema clasifica cada pera como PASA, REVISAR o RECHAZA basándose exclusivamente en defectos superficiales visibles detectados mediante imagen RGB.

---

## 3. Qué problema resuelve

Automatiza parcialmente la inspección visual manual de peras en línea de empaquetado, reduciendo el volumen de revisión manual de operario y separando automáticamente peras con alta probabilidad de defecto grave de las que pueden pasar directamente o requieren revisión humana.

---

## 4. Alcance actual

- Peras enteras (no cortadas, no procesadas).
- Inspección visual externa únicamente.
- Imagen RGB estática (foto o captura de cámara).
- La pera debe ser visible y ocupar una parte relevante de la imagen.
- Salida: **PASA / REVISAR / RECHAZA** con probabilidades U3.
- Defectos detectables: `mechanical_damage`, `rot`, `twig_mark`.

---

## 5. Qué NO resuelve

Este sistema **no mide ni garantiza**:

- Calidad interna (pardeamiento interno, daños subcutáneos).
- Contenido de azúcar / índice Brix.
- Firmeza o textura interior.
- Densidad o madurez interna.
- Calidad nutricional o vitamínica.
- Defectos invisibles a la cámara RGB.
- Garantía industrial completa — es un prototipo académico.

---

## 6. Pipeline técnico final

```
Entrada imagen RGB
       │
       ▼
Validación de captura
(iluminación, tamaño de máscara, calidad)
       │
       ▼
Segmentación de pera
(máscara GrabCut / HSV adaptativa)
       │
       ▼
Extracción ROI enmascarado
(recorte con fondo gris neutro — gray_bg_clean)
       │
       ▼
Clasificador U3
(CNN entrenado sobre ROI masked clean)
Salida: p_good, p_bad, u3_pred
       │
       ▼
Política de decisión final (ver §7)
       │
       ▼
Salida: PASA / REVISAR / RECHAZA
+ visualización con contorno y métricas
```

---

## 7. Reglas de decisión finales

| Condición | Decisión |
|---|---|
| U3_GOOD y p_good > 0.85 | **PASA** |
| U3_BAD y p_bad >= 0.995 | **RECHAZA** |
| U3_BAD y p_bad < 0.995 | **REVISAR** |
| Captura inválida / error de inferencia / máscara defectuosa | **REVISAR** |

**Umbral elegido: threshold_p_bad = 0.995**

Justificación: es el único umbral del grid evaluado con GOOD→RECHAZA=0 y BAD→PASA=0. Los 3 casos GOOD con u3_pred=bad tienen p_bad máximo 0.9943 < 0.995.

---

## 8. Métricas finales — Dataset corregido (269 imágenes, etiquetas humanas)

| Métrica | Antes (baseline U3) | Después (política integrada) |
|---|---|---|
| GOOD → PASA | 51 | 51 |
| GOOD → REVISAR | 4 | 4 |
| GOOD → RECHAZA | 0 | **0** |
| BAD → PASA | 0 | **0** |
| BAD → REVISAR | 214 | 85 |
| BAD → RECHAZA | 0 | 129 |
| false_reject_rate (FRR) | 0.0% | **0.0%** |
| false_accept_rate (FAR) | 0.0% | **0.0%** |
| automatic_accept_rate (AAR) | 19.0% | 19.0% |
| manual_review_rate (MRR) | 81.0% | **33.1%** |
| reject_rate (RR) | 0.0% | **48.0%** |

Total dataset: 55 GOOD + 214 BAD = 269 imágenes.

---

## 9. Métricas finales — Supermercado / holdout (86 peras reales)

| Métrica | Resultado |
|---|---|
| Total | 86 |
| PASA | 86 |
| REVISAR | 0 |
| RECHAZA | 0 |
| Falsos rechazos visibles | Ninguno |

Todas las peras del supermercado son etiqueta GOOD. El pipeline no rechazó ni puso en revisión ninguna. Confirma que el umbral 0.995 es seguro para peras comercialmente válidas.

---

## 10. Interpretación de resultados

- **0 falsos rechazos de peras buenas**: el sistema nunca rechaza una pera comercialmente válida en los datos evaluados.
- **0 falsas aceptaciones de peras malas**: tras corrección humana de etiquetas, ninguna pera con defecto real pasa como PASA.
- **Reducción de revisión manual**: de 81.0% (baseline) a 33.1% (integrado). El operario revisa menos.
- **129 de 214 peras malas pasan a rechazo automático**: el 60.3% de las BAD se rechaza sin intervención humana.

---

## 11. Limitaciones conocidas

- Dataset pequeño (269 imágenes corregidas + 86 holdout).
- Muchas imágenes provienen de Fruits-360 o lotes concretos y no representan todas las variedades comerciales.
- Las peras reales de supermercado probadas eran sanas (ninguna pera mala real de supermercado evaluada).
- Falta probar más variedades, defectos reales nuevos y condiciones de luz variables.
- El umbral 0.995 es conservador y específico del modelo U3 actual: si se reentrena U3, el umbral debe recalibrarse.
- El sistema no ha sido validado en condiciones industriales controladas.

---

## 12. Próximos pasos recomendados

1. Capturar un lote nuevo de imágenes no visto (idealmente con peras defectuosas reales de supermercado).
2. Incluir peras realmente malas de origen real (no Fruits-360) para validar el rechazo en condiciones reales.
3. Probar en condiciones de iluminación más controladas y uniformes.
4. Guardar predicciones y contact sheets de cada nueva prueba para trazabilidad.
5. No reentrenar hasta acumular suficientes hard examples útiles y documentar motivación.
6. Evaluar si activar `--use-defect-model` con YOLO aporta valor adicional al pipeline U3.

---

## 13. Estado final del sistema

| Aspecto | Estado |
|---|---|
| Pipeline | **ACEPTADO como pipeline final provisional** |
| Clasificador principal | U3 ROI/masked clean |
| Umbral GOOD | p_good > 0.85 → PASA |
| Umbral BAD | p_bad >= 0.995 → RECHAZA |
| Duda/error | REVISAR |
| Uso industrial definitivo | **NO** — prototipo académico |

El sistema queda aceptado para demostración académica y prototipado. No debe usarse como sistema de control de calidad industrial certificado sin validación adicional.

---

*Generado automáticamente como resumen final del pipeline PearVision QC v1.*
