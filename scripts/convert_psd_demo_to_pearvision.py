"""
Convierte el dataset demo de PSD al formato PearVision QC (YOLO).

Mapeo de clases:
  PSD 0 pear   -> ignorar (no es defecto)
  PSD 1 bruise -> PearVision 0 mechanical_damage
  PSD 2 stab   -> PearVision 0 mechanical_damage
  PSD 3 twig   -> PearVision 2 twig_mark
  PSD 4 tcm    -> ignorar (clase ambigua)
  PSD 5 rot    -> PearVision 1 rot

Uso:
    python scripts/convert_psd_demo_to_pearvision.py \
        --input data/raw/external/PearSurfaceDefects_original/PSD-main/datasets/Date_demo \
        --output data

    Agregar --dry-run para ver qué haría sin modificar nada.
"""
import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CLASS_MAP = {
    1: 0,  # bruise -> mechanical_damage
    2: 0,  # stab   -> mechanical_damage
    3: 2,  # twig   -> twig_mark
    5: 1,  # rot    -> rot
}
IGNORE_IDS = {0, 4}  # pear, tcm

CLASS_NAMES_PSD = {0: "pear", 1: "bruise", 2: "stab", 3: "twig", 4: "tcm", 5: "rot"}
CLASS_NAMES_PV = {0: "mechanical_damage", 1: "rot", 2: "twig_mark"}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def convert_label_lines(lines):
    """
    Convierte líneas YOLO de formato PSD a PearVision.
    Devuelve (lineas_convertidas, n_convertidas, n_ignoradas_pear, n_ignoradas_tcm, n_desconocidas).
    """
    out_lines = []
    n_conv = 0
    n_pear = 0
    n_tcm = 0
    n_unk = 0

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            cls = int(parts[0])
        except ValueError:
            n_unk += 1
            continue

        if cls == 0:
            n_pear += 1
        elif cls == 4:
            n_tcm += 1
        elif cls in CLASS_MAP:
            new_cls = CLASS_MAP[cls]
            out_lines.append(f"{new_cls} {' '.join(parts[1:])}")
            n_conv += 1
        else:
            n_unk += 1

    return out_lines, n_conv, n_pear, n_tcm, n_unk


def process_split(split, input_dir, output_dir, dry_run, stats):
    img_src = input_dir / "images" / split
    lbl_src = input_dir / "labels" / split
    img_dst = output_dir / split / "images"
    lbl_dst = output_dir / split / "labels"

    if not img_src.exists():
        print(f"  AVISO: no existe {img_src}, omitiendo split '{split}'.")
        return

    images = sorted(p for p in img_src.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)

    if not images:
        print(f"  AVISO: no hay imágenes en {img_src}.")
        return

    print(f"\n[{split.upper()}]  {len(images)} imágenes encontradas en {img_src}")

    if not dry_run:
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

    for img_path in images:
        prefix = "psd_demo_"
        dst_img_name = prefix + img_path.name
        dst_lbl_name = prefix + img_path.stem + ".txt"
        dst_img = img_dst / dst_img_name
        dst_lbl = lbl_dst / dst_lbl_name

        lbl_src_path = lbl_src / (img_path.stem + ".txt")

        # Leer y convertir label
        if lbl_src_path.exists():
            raw_lines = lbl_src_path.read_text(encoding="utf-8").strip().splitlines()
        else:
            raw_lines = []
            print(f"  AVISO: sin label para {img_path.name}, se creará .txt vacío.")

        conv_lines, n_conv, n_pear, n_tcm, n_unk = convert_label_lines(raw_lines)

        stats["imagenes"] += 1
        stats["labels"] += 1
        stats["cajas_originales"] += len(raw_lines)
        stats["cajas_convertidas"] += n_conv
        stats["cajas_pear"] += n_pear
        stats["cajas_tcm"] += n_tcm
        stats["desconocidos"] += n_unk
        stats["archivos_creados"] += 2  # imagen + label

        if dry_run:
            estado_lbl = "con defectos" if conv_lines else "sin defectos (vacío)"
            print(f"  [DRY-RUN] {img_path.name}")
            print(f"            -> imagen : {dst_img}")
            print(f"            -> label  : {dst_lbl}  [{estado_lbl}]")
            if n_pear:
                print(f"               ignoradas pear: {n_pear}")
            if n_tcm:
                print(f"               ignoradas tcm : {n_tcm}")
            if n_unk:
                print(f"               desconocidas  : {n_unk}")
        else:
            shutil.copy2(img_path, dst_img)
            label_text = "\n".join(conv_lines)
            dst_lbl.write_text(label_text, encoding="utf-8")
            estado_lbl = f"{len(conv_lines)} cajas" if conv_lines else "vacío"
            print(f"  OK  {dst_img_name}  label: {estado_lbl}")


def print_summary(stats, dry_run):
    modo = "DRY-RUN (nada fue modificado)" if dry_run else "CONVERSIÓN REAL"
    print("\n" + "=" * 60)
    print(f"RESUMEN — {modo}")
    print("=" * 60)
    print(f"  Imágenes procesadas          : {stats['imagenes']}")
    print(f"  Labels procesados            : {stats['labels']}")
    print(f"  Cajas originales leídas      : {stats['cajas_originales']}")
    print(f"  Cajas convertidas            : {stats['cajas_convertidas']}")
    print(f"  Cajas ignoradas (pear)       : {stats['cajas_pear']}")
    print(f"  Cajas ignoradas (tcm)        : {stats['cajas_tcm']}")
    print(f"  Class_id desconocidos        : {stats['desconocidos']}")
    print(f"  Archivos de salida creados   : {stats['archivos_creados'] if not dry_run else 0}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Convierte dataset PSD demo al formato PearVision QC YOLO"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Ruta a Date_demo (debe contener images/ y labels/)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Ruta raíz de salida (ej. data/). Se escribirá en data/train/ y data/val/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Solo imprime qué haría. No crea ni modifica ningún archivo.",
    )
    args = parser.parse_args()

    input_dir = (PROJECT_ROOT / args.input).resolve()
    output_dir = (PROJECT_ROOT / args.output).resolve()

    if not input_dir.exists():
        print(f"ERROR: la carpeta de entrada no existe: {input_dir}")
        sys.exit(1)

    print(f"Entrada  : {input_dir}")
    print(f"Salida   : {output_dir}")
    print(f"Modo     : {'DRY-RUN' if args.dry_run else 'REAL'}")
    print()
    print("Mapeo de clases:")
    for psd_id, pv_id in CLASS_MAP.items():
        print(f"  PSD {psd_id} {CLASS_NAMES_PSD[psd_id]:8s} -> PearVision {pv_id} {CLASS_NAMES_PV[pv_id]}")
    for ign_id in sorted(IGNORE_IDS):
        print(f"  PSD {ign_id} {CLASS_NAMES_PSD[ign_id]:8s} -> IGNORAR")

    stats = {
        "imagenes": 0,
        "labels": 0,
        "cajas_originales": 0,
        "cajas_convertidas": 0,
        "cajas_pear": 0,
        "cajas_tcm": 0,
        "desconocidos": 0,
        "archivos_creados": 0,
    }

    for split in ("train", "val"):
        process_split(split, input_dir, output_dir, args.dry_run, stats)

    print_summary(stats, args.dry_run)

    if not args.dry_run:
        print("\nPróximo paso recomendado:")
        print("  python scripts/check_yolo_dataset.py --data configs/yolo_pearvision.yaml")


if __name__ == "__main__":
    main()
