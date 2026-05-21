"""
Evalúa el clasificador pear_gate sobre el split test (o val si test no está disponible).

Uso:
    python scripts/evaluate_pear_gate.py \
        --weights outputs/pear_gate_train/pear_gate_yolov8n_cls_baseline/weights/best.pt \
        --data data/pear_gate \
        --device 0
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(
        description="PearVision QC — evalúa el clasificador pear_gate"
    )
    parser.add_argument("--weights", required=True,
                        help="Ruta a best.pt del experimento")
    parser.add_argument("--data", required=True,
                        help="Ruta al dataset pear_gate (ej. data/pear_gate)")
    parser.add_argument("--device", default="0",
                        help="Dispositivo: 0 para GPU, cpu para CPU (default: 0)")
    args = parser.parse_args()

    weights_path = (PROJECT_ROOT / args.weights).resolve()
    data_root = (PROJECT_ROOT / args.data).resolve()
    output_dir = (PROJECT_ROOT / "outputs" / "pear_gate_eval").resolve()

    if not weights_path.exists():
        print(f"ERROR: no se encuentra el fichero de pesos: {weights_path}")
        sys.exit(1)

    if not data_root.exists():
        print(f"ERROR: la carpeta del dataset no existe: {data_root}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    print(f"Pesos        : {weights_path}")
    print(f"Dataset      : {data_root}")
    print(f"Dispositivo  : {args.device}")
    print(f"Salida       : {output_dir}")

    model = YOLO(str(weights_path))

    # Intentar evaluar sobre test; si Ultralytics no lo soporta, usar val
    split_used = "test"
    try:
        metrics = model.val(
            data=str(data_root),
            split="test",
            device=args.device,
            project=str(output_dir),
            name="eval_test",
        )
    except (TypeError, Exception) as e:
        print(f"\nAVISO: evaluación sobre 'test' no disponible ({e}).")
        print("Usando split 'val' en su lugar.\n")
        split_used = "val"
        metrics = model.val(
            data=str(data_root),
            split="val",
            device=args.device,
            project=str(output_dir),
            name="eval_val",
        )

    print(f"\nEvaluación completada sobre split '{split_used}'.")
    print("\nMétricas disponibles:")

    # Intentar imprimir las métricas más habituales de clasificación
    for attr in ("top1", "top5", "fitness"):
        val = getattr(metrics, attr, None)
        if val is not None:
            print(f"  {attr:10s}: {val:.4f}")

    results_dir = output_dir / ("eval_test" if split_used == "test" else "eval_val")
    print(f"\nResultados guardados en: {results_dir}")


if __name__ == "__main__":
    main()
