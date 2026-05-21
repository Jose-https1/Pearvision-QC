"""
Ejecuta predicción con un modelo YOLOv8 entrenado para PearVision QC.
Uso:
    python scripts/predict_yolo.py \
        --weights outputs/yolo_train/pearvision_yolov8n_baseline/weights/best.pt \
        --source data/samples \
        --conf 0.25 \
        --name predict_test
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREDICT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "yolo_predict"


def main():
    parser = argparse.ArgumentParser(description="PearVision QC — predicción YOLOv8")
    parser.add_argument("--weights", required=True, help="Ruta al archivo .pt del modelo entrenado")
    parser.add_argument("--source",  required=True, help="Imagen, carpeta o video de entrada")
    parser.add_argument("--imgsz",   type=int, default=640,  help="Tamaño de imagen de inferencia")
    parser.add_argument("--conf",    type=float, default=0.25, help="Umbral mínimo de confianza")
    parser.add_argument("--device",  default="0",            help="Dispositivo: 0 (GPU), cpu")
    parser.add_argument("--name",    default="predict_test", help="Nombre de la carpeta de salida")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"ERROR: no se encontró el modelo: {weights_path}")
        sys.exit(1)

    source_path = Path(args.source)
    if not source_path.exists():
        print(f"ERROR: no se encontró la fuente: {source_path}")
        sys.exit(1)

    PREDICT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics no está instalado en el entorno activo.")
        print("Activa .venv_yolo antes de ejecutar este script.")
        sys.exit(1)

    print(f"Modelo   : {weights_path}")
    print(f"Fuente   : {source_path}")
    print(f"Conf     : {args.conf}")
    print(f"Device   : {args.device}")
    print(f"Salida   : {PREDICT_OUTPUT_DIR / args.name}")
    print()

    model = YOLO(str(weights_path))
    results = model.predict(
        source=str(source_path),
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        name=args.name,
        project=str(PREDICT_OUTPUT_DIR),
        save=True,
        exist_ok=True,
    )

    detecciones_totales = sum(len(r.boxes) for r in results)
    print(f"\nPredicción completada.")
    print(f"Imágenes procesadas : {len(results)}")
    print(f"Detecciones totales : {detecciones_totales}")
    print(f"Resultados en       : {PREDICT_OUTPUT_DIR / args.name}")


if __name__ == "__main__":
    main()
