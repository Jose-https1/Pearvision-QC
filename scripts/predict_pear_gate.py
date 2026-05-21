"""
Predice con el clasificador pear_gate sobre una imagen o carpeta.

Política de decisión de tres niveles:
  PASA    — valid_pear >= pass-threshold
  RECHAZA — invalid   >= reject-threshold
  REVISAR — cualquier otro caso (zona de incertidumbre)

Uso:
    python scripts/predict_pear_gate.py \
        --weights outputs/pear_gate_train/pear_gate_yolov8n_cls_baseline/weights/best.pt \
        --source data/samples \
        --pass-threshold 0.50 \
        --reject-threshold 0.80 \
        --device 0 \
        --name predict_samples
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VALID_PEAR_CLASS = "valid_pear"
INVALID_CLASS = "invalid"


def decide(p_valid: float, p_invalid: float, pass_thr: float, reject_thr: float) -> str:
    if p_valid >= pass_thr:
        return "PASA"
    if p_invalid >= reject_thr:
        return "RECHAZA"
    return "REVISAR"


def main():
    parser = argparse.ArgumentParser(
        description="PearVision QC — predicción con clasificador pear_gate (3 niveles)"
    )
    parser.add_argument("--weights", required=True,
                        help="Ruta a best.pt del experimento")
    parser.add_argument("--source", required=True,
                        help="Imagen o carpeta de imágenes a predecir")
    parser.add_argument("--pass-threshold", type=float, default=0.50,
                        help="Confianza mínima en valid_pear para PASA (default: 0.50)")
    parser.add_argument("--reject-threshold", type=float, default=0.80,
                        help="Confianza mínima en invalid para RECHAZA (default: 0.80)")
    parser.add_argument("--conf", type=float, default=None,
                        help="Alias de --pass-threshold (compatibilidad con comandos anteriores)")
    parser.add_argument("--device", default="0",
                        help="Dispositivo: 0 para GPU, cpu para CPU (default: 0)")
    parser.add_argument("--name", default="predict_samples",
                        help="Nombre de la carpeta de resultados (default: predict_samples)")
    args = parser.parse_args()

    # --conf actúa como alias de --pass-threshold si no se pasó --pass-threshold explícitamente
    pass_thr = args.pass_threshold
    if args.conf is not None:
        pass_thr = args.conf
        print(f"AVISO: --conf es un alias de --pass-threshold. Usando pass-threshold={pass_thr}")
    reject_thr = args.reject_threshold

    weights_path = (PROJECT_ROOT / args.weights).resolve()
    source_path = (PROJECT_ROOT / args.source).resolve()
    output_dir = (PROJECT_ROOT / "outputs" / "pear_gate_predict").resolve()

    if not weights_path.exists():
        print(f"ERROR: no se encuentra el fichero de pesos: {weights_path}")
        sys.exit(1)

    if not source_path.exists():
        print(f"ERROR: la fuente no existe: {source_path}")
        sys.exit(1)

    from ultralytics import YOLO

    print(f"Pesos            : {weights_path}")
    print(f"Fuente           : {source_path}")
    print(f"pass-threshold   : >= {pass_thr}")
    print(f"reject-threshold : >= {reject_thr}")
    print(f"Dispositivo      : {args.device}")
    print(f"Salida           : {output_dir / args.name}")

    model = YOLO(str(weights_path))

    # Localizar índices de las dos clases en el modelo
    name_to_idx = {v: k for k, v in model.names.items()}
    idx_valid = name_to_idx.get(VALID_PEAR_CLASS)
    idx_invalid = name_to_idx.get(INVALID_CLASS)

    if idx_valid is None or idx_invalid is None:
        print(f"ERROR: el modelo no contiene las clases esperadas.")
        print(f"  Clases del modelo: {list(model.names.values())}")
        print(f"  Se esperaban: '{VALID_PEAR_CLASS}' y '{INVALID_CLASS}'")
        sys.exit(1)

    results = model.predict(
        source=str(source_path),
        device=args.device,
        project=str(output_dir),
        name=args.name,
        save=True,
    )

    # Cabecera de tabla
    col_img     = 32
    col_top     = 12
    col_conf    = 7
    col_pvalid  = 8
    col_pinv    = 9
    col_dec     = 8
    sep = "─" * (col_img + col_top + col_conf + col_pvalid + col_pinv + col_dec + 12)

    print(f"\n{sep}")
    print(
        f"  {'IMAGEN':<{col_img}} {'CLASE_TOP':<{col_top}} "
        f"{'CONF_TOP':>{col_conf}}  {'P_VALID':>{col_pvalid}}  "
        f"{'P_INVALID':>{col_pinv}}  DECISIÓN"
    )
    print(sep)

    n_pasa = 0
    n_revisar = 0
    n_rechaza = 0

    for r in results:
        if r.probs is None:
            continue

        probs = r.probs.data          # tensor con todas las probabilidades
        top1_idx = int(r.probs.top1)
        top1_conf = float(r.probs.top1conf)
        top1_name = model.names[top1_idx]

        p_valid  = float(probs[idx_valid])
        p_invalid = float(probs[idx_invalid])

        decision = decide(p_valid, p_invalid, pass_thr, reject_thr)
        if decision == "PASA":
            n_pasa += 1
        elif decision == "REVISAR":
            n_revisar += 1
        else:
            n_rechaza += 1

        img_name = Path(r.path).name if r.path else "desconocida"
        print(
            f"  {img_name:<{col_img}} {top1_name:<{col_top}} "
            f"{top1_conf:>{col_conf}.3f}  {p_valid:>{col_pvalid}.3f}  "
            f"{p_invalid:>{col_pinv}.3f}  {decision}"
        )

    print(sep)
    print(f"  Total PASA    : {n_pasa}")
    print(f"  Total REVISAR : {n_revisar}")
    print(f"  Total RECHAZA : {n_rechaza}")
    print(sep)
    print(f"\nResultados guardados en: {output_dir / args.name}")


if __name__ == "__main__":
    main()
