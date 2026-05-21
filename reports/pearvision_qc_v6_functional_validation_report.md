# PearVision QC V6 — Informe de Validación Funcional

**Fecha:** 2026-05-21  
**Versión del sistema:** live_camera_qc_pro_v6  
**Cámara usada:** Webcam integrada del portátil (índice 0, 1280×720 px)  
**Modo de prueba:** Tiempo real con captura en vivo (sin reentrenamiento)

---

## 1. Qué se ha probado

Se ejecutó la aplicación PearVision QC V6 en modo cámara en vivo con la webcam del portátil. Durante la sesión se presentaron ante la cámara distintas escenas para validar todos los casos de decisión relevantes:

| Escena presentada | Resultado esperado | Casos |
|---|---|---|
| Fondo blanco/gris vacío | SIN PERA | 2, 5 (parcial) |
| Pera sana entera, centrada, fondo claro | PASA | 3 |
| Pera con manchas/defectos visibles | REVISAR | 6 |
| Persona / cabeza frente a cámara | MALA CAPTURA o SIN PERA | 1, 4 |
| Escena con persona + pera fuera de plano | SIN PERA | 5 |

Se guardaron 6 evidencias completas: frame original, overlay de interfaz, máscara de segmentación, ROI procesado a 224×224 y snapshot compuesto. La latencia total por frame fue de 17–31 ms.

---

## 2. Resultados obtenidos

Todos los casos dieron el resultado correcto. Ningún caso produjo un PASA ni un RECHAZA indebido.

| # | Timestamp | Frame | Decisión final | p_good | p_bad | capture_status | mask_valid | gate_reason |
|---|---|---|---|---|---|---|---|---|
| 1 | 20260521_224050_698 | 6201 | **MALA CAPTURA** | 0.000 | 0.000 | OK | ✓ | BORDER_CUT_BLOCKED |
| 2 | 20260521_224345_648 | 266 | **SIN PERA** | 0.000 | 0.000 | SIN_PERA | ✗ | NO_VALID_MASK |
| 3 | 20260521_224350_275 | 390 | **PASA** | 0.779 | 0.221 | OK | ✓ | OK |
| 4 | 20260521_224358_034 | 585 | **MALA CAPTURA** | 0.000 | 0.000 | OK | ✓ | BORDER_CUT_BLOCKED |
| 5 | 20260521_224406_541 | 795 | **SIN PERA** | 0.000 | 0.000 | SIN_PERA | ✗ | NO_VALID_MASK |
| 6 | 20260521_224446_187 | 130 | **REVISAR** | 0.159 | 0.841 | OK | ✓ | OK |

---

## 3. Tabla detallada de evidencias

| # | Archivo base | Decisión final | p_good | p_bad | capture_status | mask_valid | Observación visual | Conclusión |
|---|---|---|---|---|---|---|---|---|
| 1 | `20260521_224050_698_f006201` | MALA CAPTURA | 0.000 | 0.000 | OK | ✓ (border_cut=True) | Persona con auriculares y boca abierta frente a la cámara; la cabeza ocupa y corta el borde del frame | CORRECTO: objeto no pera detectado con borde cortado; U3 bloqueado por gate BORDER_CUT_BLOCKED antes de inferencia |
| 2 | `20260521_224345_648_f000266` | SIN PERA | 0.000 | 0.000 | SIN_PERA | ✗ | Frame gris-blanco completamente vacío y desenfocado; ningún objeto discernible | CORRECTO: fondo vacío, sin candidatos válidos → SIN PERA |
| 3 | `20260521_224350_275_f000390` | PASA | 0.779 | 0.221 | OK | ✓ | Pera entera sobre fondo blanco/gris claro en posición lateral; superficie moteada natural sin defectos evidentes; centrada | CORRECTO: pera sana clasificada PASA — p_good=0.779 ≥ umbral 0.6; U3=GOOD |
| 4 | `20260521_224358_034_f000585` | MALA CAPTURA | 0.000 | 0.000 | OK | ✓ (border_cut=True) | Persona con mano cubriendo el rostro; mano/brazo cortado en borde derecho del frame; sat_score muy bajo (0.07) | CORRECTO: captura inválida bloqueada por BORDER_CUT_BLOCKED; no hay inferencia U3 |
| 5 | `20260521_224406_541_f000795` | SIN PERA | 0.000 | 0.000 | SIN_PERA | ✗ | Persona con brazos extendidos dominando el plano; pera pequeña apenas visible en borde inferior; máscara de área enorme (425 931 px, ~46% del frame) inválida | CORRECTO: escena confusa con persona dominante → ningún candidato geométricamente válido como pera |
| 6 | `20260521_224446_187_f000130` | REVISAR | 0.159 | 0.841 | OK | ✓ | Pera entera centrada sobre fondo blanco/gris; manchas oscuras, pintas y zonas marrones visibles en superficie | CORRECTO: pera dudosa con defecto probable — p_bad=0.841, por debajo del umbral de rechazo duro (0.995) → REVISAR para inspección manual |

---

## 4. Evidencia visual

El contact sheet completo se encuentra en:

```
outputs/live_camera_qc_pro_v6/evidence_contact_sheet_v6.jpg
```

Muestra 6 filas (una por caso), 5 columnas cada una: **Original → Overlay → Máscara → ROI 224×224 → Snapshot compuesto**. Las etiquetas de decisión están coloreadas por categoría:

- Verde: PASA
- Naranja/azul claro: REVISAR  
- Azul oscuro: MALA CAPTURA  
- Gris: SIN PERA

El CSV resumen con todos los campos cuantitativos está en:

```
outputs/live_camera_qc_pro_v6/evidence_summary_v6.csv
```

---

## 5. Por qué V6 mejora frente a V4/V5

| Aspecto | V4/V5 | V6 |
|---|---|---|
| Detección de borde cortado | Ausente o parcial | Gate `BORDER_CUT_BLOCKED` sistemático: cualquier candidato que toque el borde del frame es bloqueado antes de llegar a U3 |
| Clasificación de escenas sin pera | Básica (umbral de área) | `NO_VALID_MASK` con criterios geométricos múltiples (rectangularity, solidity, pear_area_ratio) |
| Protección frente a falsos PASA por persona | No existía | La combinación border_cut + sat_score bajo impide clasificar caras/manos como peras |
| Decisión estable vs. instantánea | Solo decisión instantánea | Campo `stable_decision` separado de `instant_decision`, apto para suavizado temporal en producción |
| Trazabilidad por frame | Parcial | Metadata JSON completa por cada frame guardado: todos los parámetros, scores, latencias y razones de gate |
| Latencia total | No documentada | 17–31 ms por frame (preprocessing + inferencia U3), compatible con uso en tiempo real a 18–30 fps |

---

## 6. Limitaciones actuales

1. **Iluminación importante:** el sistema requiere iluminación ambiente uniforme y suficiente. Sombras duras o luz de rincón degradan la segmentación y pueden producir máscaras inválidas.
2. **Fondo blanco o gris recomendado:** fondos complejos o con objetos adicionales aumentan el ruido en la segmentación por saturación (estrategia `sat`) y generan candidatos falsos.
3. **Pera completa y centrada:** si la pera está parcialmente fuera del encuadre, el gate `BORDER_CUT_BLOCKED` la rechaza correctamente como MALA CAPTURA; el usuario debe reposicionar la fruta.
4. **Revisión manual si hay duda:** el nivel REVISAR (p_bad entre umbral bueno y umbral de rechazo duro) está diseñado para que un operario tome la decisión final; el sistema no clasifica automáticamente como RECHAZA salvo con certeza muy alta (p_bad ≥ 0.995).
5. **Sin calibración de fondo activa:** `bg_calibrated=False` en todos los casos. Con calibración activa del fondo específico del entorno, la segmentación mejoraría en condiciones de iluminación variable.
6. **Solo defectos superficiales visibles:** el sistema no detecta ni promete medir nada interno (firmeza, Brix, pardeamiento interno, densidad).

---

## 7. Conclusión final

**PearVision QC V6 es válida como demo funcional provisional en tiempo real.**

Los 6 casos de validación cubren los escenarios críticos definidos en la especificación del proyecto:

- Fondo vacío → SIN PERA (sin falso positivo)
- Pera sana → PASA (decisión correcta con p_good=0.78)
- Pera dudosa → REVISAR (p_bad=0.84, bajo el umbral de rechazo duro)
- Persona/mano/cabeza → MALA CAPTURA o SIN PERA (nunca PASA ni RECHAZA)
- Escena confusa con objeto no pera → SIN PERA o MALA CAPTURA (nunca PASA)

Ningún caso produjo una clasificación indebida. El pipeline completo opera en menos de 31 ms por frame, compatible con uso interactivo. El sistema puede usarse como demostración funcional del concepto PearVision QC ante la comunidad académica, con las limitaciones de iluminación y posicionamiento documentadas anteriormente.

---

*Generado automáticamente a partir de las evidencias guardadas en `outputs/live_camera_qc_pro_v6/`. No se ha reentrenado ningún modelo ni modificado ningún script de producción para la generación de este informe.*
