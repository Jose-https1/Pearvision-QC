"""
Prepara el dataset ECLPOD para entrenar un detector YOLO de pera.

Solo conserva la clase 0 (pear body) y genera split train/val/test
con las rutas que espera Ultralytics YOLO.
"""

import argparse
import random
import shutil
import sys
import py_compile
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
PEAR_CLASS_ID = 0


def parse_args():
    p = argparse.ArgumentParser(description="Prepara ECLPOD como dataset YOLO mono-clase (pear body).")
    p.add_argument("--source",       default="data_external/ECLPOD",         help="Raíz del dataset ECLPOD")
    p.add_argument("--output",       default="data/pear_detector_eclpod",     help="Carpeta de salida")
    p.add_argument("--yaml-out",     default="configs/eclpod_pear_detector.yaml", help="Ruta del YAML de entrenamiento")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--train-ratio",  type=float, default=0.8)
    p.add_argument("--val-ratio",    type=float, default=0.1)
    p.add_argument("--clear-output", action="store_true", help="Borra la carpeta de salida antes de empezar")
    return p.parse_args()


def find_pairs(src: Path):
    """Devuelve lista de (img_path, lbl_path) con correspondencia por nombre base."""
    img_dir = src / "images"
    lbl_dir = src / "labels"
    pairs = []
    for img in img_dir.iterdir():
        if img.suffix not in IMAGE_EXTENSIONS:
            continue
        lbl = lbl_dir / (img.stem + ".txt")
        if lbl.exists():
            pairs.append((img, lbl))
    return pairs


def filter_label(lbl_path: Path):
    """
    Lee un label YOLO y conserva solo las líneas de clase 0.
    Devuelve lista de líneas, o None si no hay ninguna línea de clase 0.
    """
    lines = []
    for raw in lbl_path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split()
        if int(parts[0]) == PEAR_CLASS_ID:
            # Reescribir con clase 0 (ya lo es, pero normalizamos formato)
            lines.append(" ".join(["0"] + parts[1:]))
    return lines if lines else None


def split(pairs, train_ratio, val_ratio, seed):
    rng = random.Random(seed)
    shuffled = pairs[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]
    return train, val, test


def copy_split(subset, out_root: Path, split_name: str):
    img_out = out_root / "images" / split_name
    lbl_out = out_root / "labels" / split_name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    for img_path, filtered_lines in subset:
        shutil.copy2(img_path, img_out / img_path.name)
        (lbl_out / (img_path.stem + ".txt")).write_text(
            "\n".join(filtered_lines) + "\n", encoding="utf-8"
        )


def write_yaml(out_root: Path, yaml_out: Path):
    # Escrito a mano para no depender de pyyaml en el entorno base
    content = (
        f"path: {out_root.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 1\n"
        "names:\n"
        "  0: pear\n"
    )
    yaml_out.parent.mkdir(parents=True, exist_ok=True)
    yaml_out.write_text(content, encoding="utf-8")


def verify(out_root: Path):
    ok = True
    for split_name in ("train", "val", "test"):
        imgs = list((out_root / "images" / split_name).glob("*"))
        lbls = list((out_root / "labels" / split_name).glob("*.txt"))
        print(f"  {split_name:5s}: {len(imgs):4d} imágenes, {len(lbls):4d} labels")
        # Verificar que todos los labels tienen solo clase 0
        bad = []
        for lbl in lbls:
            for line in lbl.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("0 "):
                    bad.append(lbl.name)
                    break
        if bad:
            print(f"    ERROR: labels con clase != 0: {bad[:5]}")
            ok = False
    return ok


def main():
    args = parse_args()
    src = Path(args.source)
    out = Path(args.output)
    yaml_out = Path(args.yaml_out)

    if not src.exists():
        sys.exit(f"ERROR: No existe el directorio fuente: {src}")

    if args.clear_output and out.exists():
        print(f"Borrando {out} ...")
        shutil.rmtree(out)

    print("Buscando pares imagen/label ...")
    all_pairs = find_pairs(src)
    print(f"  Pares encontrados: {len(all_pairs)}")

    print("Filtrando — conservando solo clase 0 (pear body) ...")
    valid = []
    for img, lbl in all_pairs:
        lines = filter_label(lbl)
        if lines is not None:
            valid.append((img, lines))
    print(f"  Pares con al menos una anotación clase 0: {len(valid)}")

    print(f"Creando split (seed={args.seed}, train={args.train_ratio}, val={args.val_ratio}) ...")
    train, val, test = split(valid, args.train_ratio, args.val_ratio, args.seed)
    print(f"  train={len(train)}  val={len(val)}  test={len(test)}")

    print(f"Copiando a {out} ...")
    copy_split(train, out, "train")
    copy_split(val,   out, "val")
    copy_split(test,  out, "test")

    print(f"Escribiendo YAML en {yaml_out} ...")
    write_yaml(out, yaml_out)

    print("\nVerificación:")
    ok = verify(out)
    if ok:
        print("  OK — solo clase 0 en todos los labels.")
    else:
        sys.exit("ERROR en verificación.")

    print(f"\nDataset listo en:  {out}")
    print(f"YAML de training:  {yaml_out}")
    print("\nSiguiente comando sugerido:")
    print(f"  yolo detect train model=yolov8n.pt data={yaml_out} epochs=30 imgsz=640 project=runs/pear_detector name=eclpod_v1")


if __name__ == "__main__":
    main()
