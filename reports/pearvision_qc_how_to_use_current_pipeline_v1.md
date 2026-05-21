# PearVision QC — Cómo usar el pipeline actual v1

**Fecha:** 2026-05-21

---

## Dónde poner las imágenes nuevas

Copia las imágenes que quieras analizar en:

```
data/samples/
```

O en cualquier subcarpeta dentro de `data/`. El script acepta `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`.

---

## Comandos de uso

Todos los comandos se ejecutan desde la raíz del proyecto (la carpeta que contiene `scripts/`, `data/`, etc.).

### Analizar una sola imagen

```bash
python scripts/analyze_quality.py --image data/samples/pear_01.jpg --show
```

- `--show`: muestra la visualización en pantalla.

### Analizar una carpeta completa y guardar resultados

```bash
python scripts/analyze_quality.py --source data/samples --save
```

- `--save`: guarda las imágenes anotadas en `outputs/`.

### Con reglas personalizadas

```bash
python scripts/analyze_quality.py --source data/samples --rules configs/quality_rules.yaml --save
```

### Con detector YOLO para recorte previo (opcional)

```bash
python scripts/analyze_quality.py --source data/samples --save --use-detector --detect-conf 0.50
```

> **Nota:** El detector YOLO es opcional. El pipeline base funciona sin él mediante segmentación adaptativa.

---

## Dónde mirar los resultados

Después de ejecutar con `--save`, los resultados aparecen en:

```
outputs/
```

Archivos generados típicos:
- Imágenes anotadas con contorno, decisión y métricas.
- CSV con predicciones si el script genera sumario.
- `summary.txt` con métricas globales (cuando se evalúa un lote).

---

## Cómo interpretar PASA / REVISAR / RECHAZA

| Decisión | Significado | Acción recomendada |
|---|---|---|
| **PASA** | U3 predice pera sin defectos (p_good > 0.85). | La pera puede seguir a envasado. |
| **REVISAR** | U3 predice defecto pero con baja confianza, o captura dudosa, o error de inferencia. | Un operario humano debe revisar la pera antes de decidir. |
| **RECHAZA** | U3 predice defecto con alta confianza (p_bad >= 0.995). | Retirar la pera del lote. No envasar. |

---

## Qué hacer si aparece REVISAR

1. Revisar la imagen manualmente para confirmar si hay defecto visible.
2. Si el defecto es claro → retirar la pera.
3. Si la pera parece sana → puede ser un falso positivo de baja confianza; el operario decide.
4. Guardar el caso como ejemplo para futura revisión del dataset (no reentrenar inmediatamente).

---

## Qué hacer si aparece RECHAZA

1. La decisión es automática y conservadora (p_bad >= 0.995).
2. Retirar la pera del lote sin revisión adicional (alta confianza del modelo).
3. Si hay duda sobre el rechazo, revisar la imagen de salida anotada para ver qué detectó el sistema.
4. Documentar el caso si parece un error del sistema.

---

## Qué NO hacer

- **No reentrenar automáticamente** el modelo U3 con cada prueba nueva. El reentrenamiento requiere curación manual de datos, auditoría y validación formal.
- **No modificar** `scripts/analyze_quality.py` ni `configs/quality_rules.yaml` sin proponer un plan previo.
- **No asumir** que RECHAZA es error del sistema sin revisar la imagen anotada primero.
- **No usar** este sistema como control de calidad industrial certificado — es un prototipo académico.

---

## Backup antes de cualquier cambio

Si se necesita modificar el pipeline, crear un backup antes:

```bash
cp scripts/analyze_quality.py scripts/analyze_quality_backup_YYYYMMDD.py
```

El último backup disponible es:
`scripts/analyze_quality_backup_before_u3_bad_reject_policy_20260521_122336.py`

---

*Documento de uso del pipeline PearVision QC v1 — 2026-05-21*
