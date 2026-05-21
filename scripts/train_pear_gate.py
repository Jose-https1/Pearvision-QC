"""
Entrena el clasificador binario pear_gate con YOLO Classification (Ultralytics).

Uso:
    python scripts/train_pear_gate.py \
        --data data/pear_gate \
        --model yolov8n-cls.pt \
        --epochs 20 \
        --imgsz 224 \
        --batch 64 \
        --device 0 \
        --name pear_gate_yolov8n_cls_baseline
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REQUIRED_SUBDIRS = [
    "train/valid_pear",
    "train/invalid",
    "val/valid_pear",
    "val/invalid",
    "test/valid_pear",
    "test/invalid",
]


def verify_dataset(data_root: Path):
    if not data_root.exists():
        print(f"ERROR: la carpeta del dataset no existe: {data_root}")
        sys.exit(1)
    missing = [s for s in REQUIRED_SUBDIRS if not (data_root / s).exists()]
    if missing:
        print("ERROR: faltan las siguientes subcarpetas en el dataset:")
        for m in missing:
            print(f"  {m}")
        sys.exit(1)
    print(f"Dataset verificado: {data_root}")
    for s in REQUIRED_SUBDIRS:
        folder = data_root / s
        n = sum(1 for p in folder.iterdir() if p.is_file())
        print(f"  {s:25s}: {n} imágenes")


def main():
    parser = argparse.ArgumentParser(
        description="PearVision QC — entrena clasificador pear_gate (YOLO Classification)"
    )
    parser.add_argument("--data", required=True,
                        help="Ruta al dataset pear_gate (ej. data/pear_gate)")
    parser.add_argument("--model", default="yolov8n-cls.pt",
                        help="Modelo base (default: yolov8n-cls.pt)")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Épocas de entrenamiento (default: 20)")
    parser.add_argument("--imgsz", type=int, default=224,
                        help="Tamaño de imagen (default: 224)")
    parser.add_argument("--batch", type=int, default=64,
                        help="Batch size (default: 64)")
    parser.add_argument("--device", default="0",
                        help="Dispositivo: 0 para GPU, cpu para CPU (default: 0)")
    parser.add_argument("--name", default="pear_gate_yolov8n_cls_baseline",
                        help="Nombre del experimento (default: pear_gate_yolov8n_cls_baseline)")
    args = parser.parse_args()

    data_root = (PROJECT_ROOT / args.data).resolve()
    output_project = (PROJECT_ROOT / "outputs" / "pear_gate_train").resolve()

    # Verificar dataset antes de importar Ultralytics (fallo rápido)
    verify_dataset(data_root)

    from ultralytics import YOLO

    print(f"\nModelo base  : {args.model}")
    print(f"Épocas       : {args.epochs}")
    print(f"Imagen       : {args.imgsz}x{args.imgsz}")
    print(f"Batch        : {args.batch}")
    print(f"Dispositivo  : {args.device}")
    print(f"Experimento  : {args.name}")
    print(f"Salida       : {output_project / args.name}")

    model = YOLO(args.model)

    results = model.train(
        data=str(data_root),
        task="classify",
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(output_project),
        name=args.name,
    )

    best_pt = output_project / args.name / "weights" / "best.pt"
    print(f"\nEntrenamiento completado.")
    print(f"Experimento  : {output_project / args.name}")
    print(f"Mejor modelo : {best_pt}")
    if not best_pt.exists():
        print("  AVISO: best.pt no encontrado en la ruta esperada. Revisa la carpeta del experimento.")


if __name__ == "__main__":
    main()
