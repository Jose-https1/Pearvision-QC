# Reporte: PSD -> Custom Pear Defects

**Fecha:** 2026-05-17

## Resumen por split

| Split | Copiadas | Descartadas (sin defecto) | Errores |
|-------|---------|--------------------------|---------|
| train | 7 | 1 | 0 |
| val | 6 | 0 | 0 |
| **Total** | **13** | **1** | **0** |

## Conteo por clase (total anotaciones)

| ID | Nombre | Anotaciones |
|----|--------|------------|
| 0 | bruise | 35 |
| 1 | stab | 13 |
| 2 | twig | 0 |
| 3 | tcm | 0 |
| 4 | rot | 0 |

## Imagenes descartadas (train)

- cam0_1_6_pear_1

## Dataset listo

El dataset esta listo para entrenamiento si no hay errores criticos.

Comando de validacion:

```powershell
uv run python scripts/prepare_custom_defect_dataset.py
```