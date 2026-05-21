"""
Inspecciona un dataset externo sin modificarlo.
Detecta estructura, formatos de etiqueta y clases usadas.

Uso:
    python scripts/inspect_external_dataset.py --input data/raw/external/PearSurfaceDefects_original
"""
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CLASS_HINT_NAMES = {
    "classes.txt", "data.yaml", "dataset.yaml", "obj.names",
    "_darknet.labels", "labels.txt", "names.txt",
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _all_files(root):
    return [p for p in root.rglob("*") if p.is_file()]


def _section(title):
    print(f"\n{'-' * 60}")
    print(f"  {title}")
    print(f"{'-' * 60}")


def _show_examples(paths, n=10, label=""):
    if not paths:
        return
    print(f"\n  Ejemplos de {label} (máx. {n}):")
    for p in sorted(paths)[:n]:
        print(f"    {p}")


# ── Parsers de etiquetas ──────────────────────────────────────────────────────

def _count_yolo_classes(txt_files, root):
    """Lee archivos .txt con formato YOLO y cuenta class_id usados."""
    counter = Counter()
    malformed = 0
    for f in txt_files:
        try:
            for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                if len(parts) == 5:
                    try:
                        counter[int(parts[0])] += 1
                    except ValueError:
                        malformed += 1
                else:
                    malformed += 1
        except Exception:
            malformed += 1
    return counter, malformed


def _extract_voc_classes(xml_files):
    """Extrae nombres de clases de archivos XML Pascal VOC sin dependencias externas."""
    names = Counter()
    errors = 0
    for f in xml_files:
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
            # Búsqueda manual de <name>...</name> dentro de <object>
            pos = 0
            while True:
                obj_start = text.find("<object>", pos)
                if obj_start == -1:
                    break
                obj_end = text.find("</object>", obj_start)
                if obj_end == -1:
                    break
                chunk = text[obj_start:obj_end]
                n_start = chunk.find("<name>")
                n_end = chunk.find("</name>")
                if n_start != -1 and n_end != -1:
                    name = chunk[n_start + 6:n_end].strip()
                    if name:
                        names[name] += 1
                pos = obj_end + 1
        except Exception:
            errors += 1
    return names, errors


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PearVision QC — inspección de dataset externo (solo lectura)"
    )
    parser.add_argument("--input", required=True, help="Carpeta raíz del dataset externo")
    args = parser.parse_args()

    root = (PROJECT_ROOT / args.input).resolve()
    if not root.exists():
        print(f"ERROR: la carpeta no existe: {root}")
        sys.exit(1)

    print(f"\nInspeccionando: {root}")
    print("(modo solo lectura — no se modifica ningún archivo)\n")

    all_files = _all_files(root)

    # ── 1. Imágenes ───────────────────────────────────────────────────────────
    _section("1. IMÁGENES")
    images = [f for f in all_files if f.suffix.lower() in IMAGE_EXTENSIONS]
    ext_counter = Counter(f.suffix.lower() for f in images)
    print(f"  Total de imágenes : {len(images)}")
    print(f"  Extensiones       :")
    for ext, cnt in ext_counter.most_common():
        print(f"    {ext:10s}  {cnt}")
    _show_examples(images, label="imágenes")

    # ── 2. Archivos de etiqueta ───────────────────────────────────────────────
    _section("2. ARCHIVOS DE ETIQUETA")
    txt_files  = [f for f in all_files if f.suffix.lower() == ".txt"
                  and f.name.lower() not in CLASS_HINT_NAMES]
    xml_files  = [f for f in all_files if f.suffix.lower() == ".xml"]
    json_files = [f for f in all_files if f.suffix.lower() == ".json"]
    yaml_files = [f for f in all_files if f.suffix.lower() in {".yaml", ".yml"}]

    print(f"  Archivos .txt   : {len(txt_files)}")
    print(f"  Archivos .xml   : {len(xml_files)}")
    print(f"  Archivos .json  : {len(json_files)}")
    print(f"  Archivos .yaml  : {len(yaml_files)}")

    _show_examples(txt_files,  label="archivos .txt")
    _show_examples(xml_files,  label="archivos .xml")
    _show_examples(json_files, label="archivos .json")
    _show_examples(yaml_files, label="archivos .yaml")

    # ── 3. Estructura de carpetas relevantes ──────────────────────────────────
    _section("3. CARPETAS RELEVANTES")
    all_dirs = sorted({f.parent for f in all_files})

    images_dirs = [d for d in all_dirs if d.name.lower() in {"images", "imgs", "image"}]
    labels_dirs = [d for d in all_dirs if d.name.lower() in {"labels", "label", "annotations", "ann"}]

    print(f"  Posibles carpetas 'images' ({len(images_dirs)}):")
    for d in images_dirs:
        print(f"    {d.relative_to(root)}")

    print(f"  Posibles carpetas 'labels' ({len(labels_dirs)}):")
    for d in labels_dirs:
        print(f"    {d.relative_to(root)}")

    # ── 4. Archivos de configuración de clases ────────────────────────────────
    _section("4. ARCHIVOS DE CONFIGURACIÓN DE CLASES")
    hint_files = [
        f for f in all_files if f.name.lower() in CLASS_HINT_NAMES
    ]
    if hint_files:
        print(f"  Encontrados ({len(hint_files)}):")
        for f in hint_files:
            print(f"\n  -- {f.relative_to(root)} --")
            try:
                content = f.read_text(encoding="utf-8", errors="ignore").strip()
                for line in content.splitlines()[:30]:
                    print(f"    {line}")
            except Exception as e:
                print(f"    (error al leer: {e})")
    else:
        print("  No se encontraron archivos de configuración de clases conocidos.")

    # ── 5. Análisis de etiquetas YOLO (.txt) ──────────────────────────────────
    _section("5. ANÁLISIS DE ETIQUETAS YOLO (.txt)")
    if txt_files:
        class_counter, malformed = _count_yolo_classes(txt_files, root)
        if class_counter:
            print(f"  class_id encontrados en archivos .txt:")
            for cls_id, cnt in sorted(class_counter.items()):
                print(f"    class_id {cls_id:3d}  →  {cnt} anotación(es)")
            print(f"\n  Líneas con formato incorrecto: {malformed}")
            print()
            print("  REFERENCIA PearVision QC:")
            print("    0 → mechanical_damage")
            print("    1 → rot")
            print("    2 → twig_mark")
        else:
            print("  Los archivos .txt no parecen contener etiquetas YOLO válidas.")
            print(f"  Líneas con formato incorrecto: {malformed}")
    else:
        print("  No hay archivos .txt de etiqueta (excluidos los de configuración).")

    # ── 6. Análisis de etiquetas Pascal VOC (.xml) ────────────────────────────
    _section("6. ANÁLISIS DE ETIQUETAS PASCAL VOC (.xml)")
    if xml_files:
        voc_classes, voc_errors = _extract_voc_classes(xml_files)
        if voc_classes:
            print(f"  Clases encontradas en XML:")
            for name, cnt in voc_classes.most_common():
                print(f"    '{name}'  →  {cnt} objeto(s)")
            if voc_errors:
                print(f"\n  Archivos XML con error de lectura: {voc_errors}")
        else:
            print("  Los archivos .xml no parecen ser Pascal VOC o están vacíos.")
    else:
        print("  No hay archivos .xml.")

    # ── 7. Archivos JSON ──────────────────────────────────────────────────────
    _section("7. ARCHIVOS JSON (solo listado)")
    if json_files:
        print(f"  Total: {len(json_files)}")
        print("  (No se parsean en profundidad — requiere inspección manual)")
        _show_examples(json_files, label="archivos .json")
    else:
        print("  No hay archivos .json.")

    # ── Resumen final ─────────────────────────────────────────────────────────
    _section("RESUMEN FINAL")
    formatos = []
    if txt_files:  formatos.append("YOLO (.txt)")
    if xml_files:  formatos.append("Pascal VOC (.xml)")
    if json_files: formatos.append("JSON")
    print(f"  Imágenes totales     : {len(images)}")
    print(f"  Formatos de etiqueta : {', '.join(formatos) if formatos else 'ninguno detectado'}")
    print(f"  Archivos de config   : {len(hint_files)}")
    print()
    print("  SIGUIENTE PASO:")
    print("  Comparte este output con Claude para definir el mapeo de clases")
    print("  antes de convertir al formato PearVision QC.")
    print()


if __name__ == "__main__":
    main()
