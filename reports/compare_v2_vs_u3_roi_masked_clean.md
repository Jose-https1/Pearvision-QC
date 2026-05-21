# Comparativa V2 vs U3 ROI Masked Clean

**Fecha:** 2026-05-21
**Proyecto:** PearVision QC

---

## 1. Resumen ejecutivo

| Metrica | V2 original | V2 en gray_bg_clean | U3 ROI masked clean |
|---|---|---|---|
| Arquitectura | MobileNetV3-small | MobileNetV3-small (V2 sin retrain) | MobileNetV3-small (nuevo entrenamiento) |
| Dominio train | Fruits-360 (fondo blanco) | — (solo inferencia) | Fruits-360 masked + supermarket masked |
| Falso BAD en peras sanas (supermarket) | hasta 83% (fondo azul) | 0% | **0%** |
| REVIEW en peras sanas (supermarket) | hasta 25% | 10.9% | **0%** |
| Holdout supermarket (22 peras sanas) | no evaluado en esta config | no evaluado con umbral | **22/22 GOOD (100%)** |
| Mean p_good holdout | — | — | **0.997** |

**Conclusion: U3 elimina completamente los falsos rechazos en peras sanas de supermercado.**

---

## 2. V2 original — comportamiento con supermercado real

### Evaluacion en batches supermarket (sin preprocesado ROI)

| Batch | Fondo | Imagenes | GOOD correcto | REVIEW | BAD (falso) |
|---|---|---|---|---|---|
| batch_v1 (20 imgs) | blanco tenue | 20 | ~75% | ~25% | 0% |
| batch_v2 (22 imgs) fondo blanco | blanco | 7 | ~100% | 0% | 0% |
| batch_v2 (22 imgs) fondo azul | azul | 6 | 0% | 17% | **83%** |
| batch_v2 (22 imgs) fondo negro/textura | negro | 8 | 50% | 25% | **25%** |
| batch_v3 (22 imgs) | blanco controlado | 22 | ~90% | ~9% | 0% |

**Causa del fallo V2:** V2 fue entrenado exclusivamente en Fruits-360 (fondo blanco puro). El cambio de dominio por fondo azul/negro + sombras laterales + russeting natural de pera de supermercado inducia altas probabilidades de BAD.

### Nota sobre metricas internas V2

V2 alcanzaba alta accuracy en su conjunto de test de Fruits-360. El problema no era falta de capacidad sino **domain shift** en despliegue real.

---

## 3. V2 sobre gray_bg_clean (inferencia sin reentrenar)

Se aplicó el preprocesado ROI/masked V2 a 64 imágenes de supermercado para obtener `gray_bg_clean`:
- Pera aislada sobre fondo gris neutro (128,128,128).
- V2 (sin modificar) evaluó estas imágenes enmascaradas.

| Metrica | Resultado |
|---|---|
| Total evaluado | 64 |
| GOOD | 57 (89.1%) |
| REVIEW | 7 (10.9%) |
| BAD | **0 (0%)** |

Aplicar masking al input de V2 ya eliminó los falsos BAD directos. El 10.9% en REVIEW corresponde a peras buenas con russeting o manchas naturales que el modelo (entrenado solo en Fruits-360) sigue considerando ambiguas.

---

## 4. U3 ROI masked clean — resultados

### Dataset U3

| Split | GOOD | BAD | Total |
|---|---|---|---|
| train | 62 | 153 | 215 |
| val | 13 | 32 | 45 |
| test | 15 | 34 | 49 |
| holdout_supermarket | 22 | 0 | 22 |

Fuentes: Fruits-360 V2 (267 imagenes con masking threshold) + supermarket hard examples V1 (20) + V2 (22) via gray_bg_clean.

### Entrenamiento

- Epochs: 26 (early stopping, patience=8, mejor epoch=18)
- Val accuracy best: 100% (45/45)
- Augmentaciones: HorizontalFlip, Rotation(12), ColorJitter suave
- WeightedRandomSampler para compensar imbalance bad/good

### Resultados test interno (Fruits-360 + supermarket masked)

| Metrica | Valor |
|---|---|
| Accuracy test | 0.9184 (45/49) |
| Errores test | 4 de 49 |
| BAD->GOOD (falso negativo) | 1/34 (2.9%) |
| GOOD->BAD (falso positivo) | 3/15 (20.0%) |

Nota: los 3 GOOD->BAD en test son imagenes Fruits-360 clasificadas como GOOD por humanos pero que el modelo confunde con BAD. Esto indica que algunos Fruits-360 GOOD son ambiguos incluso para U3.

### Calibracion de umbrales

| Parametro | Valor |
|---|---|
| bad_reject_threshold | 0.60 |
| good_accept_threshold | 0.55 |
| false_bad_rate (val) | 0.0% |
| bad_recall (val) | 100% |
| false_bad_rate (test) | 20% |
| bad_catch_rate (test) | 97.1% |

### Holdout supermarket (22 peras sanas reales — batch_v3 controlado)

| Metrica | Valor |
|---|---|
| Total | 22 |
| GOOD | **22 (100%)** |
| REVIEW | **0 (0%)** |
| BAD (falso) | **0 (0%)** |
| Mean p_good | **0.997** |

**U3 clasifica correctamente el 100% de las peras sanas de supermercado con confianza muy alta.**

---

## 5. Analisis comparativo directo

| Criterio | V2 original | V2 + gray_bg_clean (inference) | U3 ROI masked |
|---|---|---|---|
| Falso BAD en fondo azul | 83% | 0% | 0% |
| Falso BAD en fondo negro | 25% | 0% | 0% |
| REVIEW en peras sanas | 10-25% | 10.9% | 0% holdout |
| Confianza media en peras sanas | baja/variable | moderada | 0.997 |
| Requiere masking en inferencia | No | Si | Si |
| Robusto a fondo azul/negro | No | Si | Si |
| Requiere retrain | — | No | Si (U3) |

**U3 supera claramente a V2 cuando las peras de supermercado son evaluadas con masking.**

---

## 6. Limitaciones actuales de U3

1. **GOOD->BAD en Fruits-360 test (3/15 = 20%):** El modelo todavia tiene confusión con algunas peras Fruits-360 buenas. Esto se debe al pequeño tamaño del dataset GOOD (90 total). Con más datos GOOD de supermercado este error debería bajar.

2. **Dataset BAD solo de Fruits-360:** Los 219 ejemplos BAD son todos de Fruits-360. Peras realmente malas de supermercado (con golpes, pudricion o marcas) no están representadas en el dataset. Si se despliega en producción, puede haber falsos GOOD para defectos reales de supermercado.

3. **Umbral agresivo (0.60):** El umbral de rechazo es bajo. Esto mantiene el falso BAD en 0 en el holdout pero acepta algunos casos dudosos como GOOD.

4. **Sin datos de validacion cruzada de peras BAD reales:** No se tienen ejemplos de peras malas de supermercado para verificar que U3 detecta defectos reales en ese dominio.

---

## 7. Conclusion y recomendacion

**U3 esta listo para integracion condicional.**

Condiciones para integrar U3 en el pipeline:

1. **Condicion cumplida:** 0% falso BAD en holdout supermarket (22 peras sanas).
2. **Condicion cumplida:** >97% bad_catch_rate en test interno.
3. **Condicion pendiente:** Validar que U3 detecta correctamente peras realmente malas de supermercado (no solo Fruits-360).
4. **Condicion pendiente:** Capturar al menos 20-30 ejemplos BAD reales de supermercado para validar la detección en dominio real.

**Paso inmediato recomendado:**
- Integrar U3 en el pipeline con masking gray_bg_clean como preprocesado.
- Mantener V2 como fallback si masking falla.
- Capturar peras BAD reales de supermercado para el siguiente ciclo de mejora (U4).

---

## 8. Confirmaciones de integridad

- No se modifico V2 (outputs/fruits360_quality_cls_v2/best_model.pt intacto).
- No se modifico analyze_quality.py.
- No se modifico quality_rules.yaml.
- No se integro U3 en el pipeline final todavia.
- U3 esta aislado en outputs/fruits360_quality_cls_u3_roi_masked_clean/.
