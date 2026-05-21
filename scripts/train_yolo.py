"""
Entrena YOLOv8 para PearVision QC.
Uso mínimo:
    python scripts/train_yolo.py --data configs/yolo_pearvision.yaml

Uso completo:
    python scripts/train_yolo.py \
        --data configs/yolo_pearvision.yaml \
        --model yolov8n.pt \
        --epochs 50 \
        --imgsz 640 \
        --batch 8 \
        --device 0 \
        --name pearvision_yolov8n_baseline
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "yolo_train"


def main():
    parser = argparse.ArgumentParser(description="PearVision QC — entrenamiento YOLOv8")
    parser.add_argument("--data",   default="configs/yolo_pearvision.yaml", help="Config YAML del dataset")
    parser.add_argument("--model",  default="yolov8n.pt",  help="Modelo base (COCO pretrained o ruta a .pt propio)")
    parser.add_argument("--epochs", type=int, default=50,  help="Número de épocas")
    parser.add_argument("--imgsz",  type=int, default=640, help="Tamaño de imagen de entrada")
    parser.add_argument("--batch",  type=int, default=8,   help="Tamaño de batch")
    parser.add_argument("--device", default="0",           help="Dispositivo: 0 (GPU), cpu")
    parser.add_argument("--name",   default="pearvision_yolov8n_baseline", help="Nombre del experimento")
    args = parser.parse_args()

    data_yaml = PROJECT_ROOT / args.data
    if not data_yaml.exists():
        print(f"ERROR: no se encontró el archivo de datos: {data_yaml}")
        sys.exit(1)

    TRAIN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics no está instalado en el entorno activo.")
        print("Activa .venv_yolo antes de ejecutar este script.")
        sys.exit(1)

    print(f"Modelo base    : {args.model}")
    print(f"Dataset config : {data_yaml}")
    print(f"Épocas         : {args.epochs}")
    print(f"Image size     : {args.imgsz}")
    print(f"Batch          : {args.batch}")
    print(f"Device         : {args.device}")
    print(f"Experimento    : {args.name}")
    print(f"Salida         : {TRAIN_OUTPUT_DIR / args.name}")
    print()

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        project=str(TRAIN_OUTPUT_DIR),
        exist_ok=True,
    )

    print(f"\nEntrenamiento completado.")
    print(f"Resultados en: {TRAIN_OUTPUT_DIR / args.name}")


if __name__ == "__main__":
    main()
