# Mendeley Good/Bad Pear — Dataset Report

## Fuente
- `C:\Users\joser\Desktop\Sistemas de Percepción y Visión Artificial\computer vision\data_external\mendeley_good_bad_pear\raw_clean\good`: 529 imagenes
- `C:\Users\joser\Desktop\Sistemas de Percepción y Visión Artificial\computer vision\data_external\mendeley_good_bad_pear\raw_clean\bad`: 500 imagenes
- Total: 1029 imagenes

## Split (seed=42)

| Split | good | bad | Total |
|-------|------|-----|-------|
| train | 370 | 350 | 720 |
| val | 105 | 100 | 205 |
| test | 54 | 50 | 104 |

## Destino
`C:\Users\joser\Desktop\Sistemas de Percepción y Visión Artificial\computer vision\data\pear_quality_cls_mendeley`

## Clases
- `good` (0 o 1 segun orden alfabetico en YOLO)
- `bad`  (0 o 1 segun orden alfabetico en YOLO)

## Notas
- Imagenes copiadas (no movidas).
- Archivos ocultos (`.trashed-*`) excluidos.
- Split estratificado por clase con `random.Random(42)`.