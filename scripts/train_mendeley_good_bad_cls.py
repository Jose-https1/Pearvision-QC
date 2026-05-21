"""
train_mendeley_good_bad_cls.py
Entrena clasificador binario good/bad sobre el dataset Mendeley preparado.

Modelo base : yolov8n-cls.pt
Epochs      : 30
imgsz       : 224
batch       : 16  (si falla por memoria, bajar a 8)
workers     : 0
cache       : False

Salida esperada:
  runs/pear_quality_cls/mendeley_good_bad_v1/weights/best.pt
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "pear_quality_cls_mendeley"


def main():
    from ultralytics import YOLO

    if not DATA_DIR.exists():
        print(f"ERROR: dataset no encontrado: {DATA_DIR}")
        print("Ejecuta primero: python scripts/prepare_mendeley_good_bad_cls.py")
        sys.exit(1)

    print("=== train_mendeley_good_bad_cls ===")
    print(f"  Dataset : {DATA_DIR}")

    model = YOLO("yolov8n-cls.pt")

    try:
        results = model.train(
            data=str(DATA_DIR),
            epochs=30,
            imgsz=224,
            batch=16,
            workers=0,
            cache=False,
            project=str(PROJECT_ROOT / "runs" / "pear_quality_cls"),
            name="mendeley_good_bad_v1",
        )
    except RuntimeError as e:
        if "memory" in str(e).lower() or "cuda" in str(e).lower():
            print(f"  Advertencia de memoria: {e}")
            print("  Reintentando con batch=8 ...")
            model = YOLO("yolov8n-cls.pt")
            results = model.train(
                data=str(DATA_DIR),
                epochs=30,
                imgsz=224,
                batch=8,
                workers=0,
                cache=False,
                project=str(PROJECT_ROOT / "runs" / "pear_quality_cls"),
                name="mendeley_good_bad_v1",
            )
        else:
            raise

    best = PROJECT_ROOT / "runs" / "pear_quality_cls" / "mendeley_good_bad_v1" / "weights" / "best.pt"
    if best.exists():
        print(f"\n  best.pt guardado en: {best}")
    else:
        print(f"\n  AVISO: best.pt no encontrado en ruta esperada: {best}")

    print("=== Entrenamiento completado ===")


if __name__ == "__main__":
    main()
