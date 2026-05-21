# PearVision QC — Auditoría del Pipeline Completo

Generado: 2026-05-17 19:47

## 1. Resumen Global

| Métrica | Valor |
|---------|-------|
| Total imágenes | 12 |
| PASA | 1 |
| REVISAR | 8 |
| RECHAZA | 3 |
| Necesitan revisión manual | 0 |

## 2. Tabla Detallada

| imagen | decision | cat_estimada | defect% | rot% | region% | yolo_valid | yolo_ignored | cls_pred | cls_bad | audit_flag |
|--------|----------|-------------|---------|------|---------|------------|--------------|----------|---------|------------|
| 1000057648.jpg | REVISAR | NO COMERCIAL | 22.44 | 0.0 | 9.85 | 0 | N/A | bad | 0.9483 | ok |
| 1000057649.jpg | RECHAZA | NO COMERCIAL | 51.27 | 4.65 | 47.01 | 0 | N/A | bad | 0.5856 | ok |
| 1000057651.jpg | PASA | CATEGORIA I | 0.74 | 0.62 | 0.74 | 0 | N/A | bad | 0.5532 | ok |
| 1000057652.jpg | REVISAR | CATEGORIA II | 2.74 | 2.58 | 0.76 | 1 | N/A | good | 0.0007 | ok |
| 1000057653.jpg | RECHAZA | NO COMERCIAL | 43.55 | 22.56 | 37.62 | 0 | N/A | good | 0.0037 | ok |
| 1000057654.jpg | REVISAR | NO COMERCIAL | 18.33 | 13.88 | 17.07 | 0 | N/A | good | 0.092 | ok |
| 1000057655.jpg | REVISAR | NO COMERCIAL | 14.2 | 3.88 | 13.29 | 0 | N/A | good | 0.1643 | ok |
| 1000057656.jpg | REVISAR | CATEGORIA II | 3.49 | 2.58 | 0.55 | 0 | N/A | good | 0.11 | ok |
| 1000057657.jpg | REVISAR | CATEGORIA II | 2.49 | 2.4 | 2.34 | 0 | N/A | good | 0.1933 | ok |
| 1000057658.jpg | REVISAR | NO COMERCIAL | 74.16 | 66.8 | 73.83 | 0 | N/A | good | 0.1404 | ok |
| 1000057659.jpg | RECHAZA | NO COMERCIAL | 99.75 | 98.51 | 99.75 | 0 | N/A | good | 0.0393 | ok |
| 1000057660.jpg | REVISAR | NO COMERCIAL | 6.67 | 6.26 | 2.98 | 0 | N/A | good | 0.0728 | ok |

## 3. Posibles Falsos Rechazos

_Ninguno detectado._

## 4. Posibles Falsos Aceptados (PASA con CLS=BAD alto)

_Ninguno detectado._

## 5. Problemas de Máscara

_Ninguno detectado._

## 6. Conclusión Técnica

### Detector de pera (YOLO ECLPOD)
- Confianza media de detección: 0.86
- Capturas válidas: 12/12
- **Evaluación**: El detector es estable — todas las imágenes fueron capturadas correctamente con confianza media 0.86.

### Máscara ROI (GrabCut)
- GrabCut exitoso: 12/12
- mask_quality_ok=True: 12/12
- **Evaluación**: La máscara GrabCut es estable en este conjunto. No se usó fallback ellipse en ninguna imagen.

### Modelo de defectos PSD (YOLO)
- Detecciones válidas totales en el conjunto: 1
- **Evaluación**: El modelo PSD detectó 1 defectos válidos. Revisar si coinciden visualmente con las zonas defectuosas reales.

### Clasificador GOOD/BAD Mendeley
- Predice GOOD: 9/12 imágenes
- Predice BAD: 3/12 imágenes
- Posibles falsos aceptados detectados: 0
- **Evaluación**: El clasificador predice GOOD en la gran mayoría de imágenes, incluso en peras con podredumbre visible (>90% de superficie afectada). Esto confirma el sesgo de dataset detectado previamente: el modelo aprendió características de composición fotográfica (screenshots vs fotos reales), no de calidad superficial de la fruta.
- **Recomendación**: Mantener el clasificador Mendeley **solo como señal informativa** (`--use-quality-cls` sin `--quality-cls-affect-decision`). No activar `--quality-cls-affect-decision` hasta reentrenar con imágenes representativas de defectos reales.

### Resumen de Acciones Recomendadas

- El pipeline muestra comportamiento coherente en este conjunto de prueba.