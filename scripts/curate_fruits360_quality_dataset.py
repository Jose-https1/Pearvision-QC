"""
curate_fruits360_quality_dataset.py
Cura el dataset Fruits-360 evaluado por el pipeline PearVision QC.

Lee:
  outputs/quality_analysis_fruits360_good_eval/resultados_calidad.csv
  data/samples_quality_fruits360_good_eval/

Crea:
  data/quality_curated_fruits360_v1/{good,bad,review,excluded_invalid_capture}/
  data/quality_curated_fruits360_v1/curated_labels.csv
  data/quality_curated_fruits360_cls_v1/{train,val,test}/{good,bad}/
  outputs/fruits360_curated_quality_preview/{good,bad,review,excluded_invalid_capture,mixed}_grid.jpg
  reports/fruits360_quality_curated_v1_report.md
  reports/fruits360_quality_curated_v1_validation.txt
"""

import csv
import math
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_CSV  = PROJECT_ROOT / "outputs" / "quality_analysis_fruits360_good_eval" / "resultados_calidad.csv"
SRC_IMGS     = PROJECT_ROOT / "data" / "samples_quality_fruits360_good_eval"

CURATED_ROOT = PROJECT_ROOT / "data" / "quality_curated_fruits360_v1"
CLS_ROOT     = PROJECT_ROOT / "data" / "quality_curated_fruits360_cls_v1"
PREVIEW_DIR  = PROJECT_ROOT / "outputs" / "fruits360_curated_quality_preview"
REPORT_MD    = PROJECT_ROOT / "reports" / "fruits360_quality_curated_v1_report.md"
VALIDATION_TXT = PROJECT_ROOT / "reports" / "fruits360_quality_curated_v1_validation.txt"

GROUPS   = ["good", "bad", "review", "excluded_invalid_capture"]
SEED     = 42
SPLIT    = {"train": 0.70, "val": 0.20, "test": 0.10}

# Columnas a exportar en curated_labels.csv
EXPORT_COLS = [
    "image", "source_path", "curated_group", "original_decision", "display_label",
    "defect_pct", "dark_rot_pct", "max_region_pct", "brown_dark_pct",
    "quality_cls_pred", "quality_cls_good_conf", "quality_cls_bad_conf",
    "yolo_defect_valid", "mask_quality_ok", "mask_fail_reason",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fl(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _bool(v):
    return str(v).strip().lower() in ("true", "1", "yes")


def _get_font(size):
    for p in ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/calibri.ttf"]:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _load_thumb(path: Path, w=140, h=130):
    try:
        buf = np.frombuffer(path.read_bytes(), dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("imdecode failed")
        scale = min(w / bgr.shape[1], h / bgr.shape[0])
        nw = max(1, int(bgr.shape[1] * scale))
        nh = max(1, int(bgr.shape[0] * scale))
        bgr = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.full((h, w, 3), 40, dtype=np.uint8)
        yo, xo = (h - nh) // 2, (w - nw) // 2
        canvas[yo:yo+nh, xo:xo+nw] = bgr
        return Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)), True
    except Exception:
        img = Image.fromarray(np.full((h, w, 3), 55, dtype=np.uint8))
        return img, False


# ---------------------------------------------------------------------------
# TAREA 1+2: Curado — asignar grupo a cada imagen
# ---------------------------------------------------------------------------

def is_invalid_capture(row: dict) -> bool:
    """Retorna True si la captura no es válida para entrenamiento."""
    if not _bool(row.get("capture_valid", "True")):
        return True
    if not _bool(row.get("mask_quality_ok", "True")):
        return True
    if row.get("mask_fail_reason", "").strip():
        return True
    if row.get("quality_cls_source", "").strip().lower() in ("not_used", "unknown", ""):
        # Captura sin crop válido
        return True
    label = row.get("display_label", "").lower()
    if "captura" in label or "detectada" in label:
        return True
    # Decisión REVISAR con todas las métricas a cero (nada se analizó)
    if row.get("decision", "") == "REVISAR":
        all_zero = (
            _fl(row.get("defect_pct", 0)) == 0.0
            and _fl(row.get("dark_rot_pct", 0)) == 0.0
            and _fl(row.get("max_region_pct", 0)) == 0.0
            and _fl(row.get("pear_visible_pct", 0)) == 0.0
        )
        if all_zero:
            return True
    return False


def assign_group(row: dict) -> str:
    if is_invalid_capture(row):
        return "excluded_invalid_capture"
    decision = row.get("decision", "")
    if decision == "PASA":
        return "good"
    if decision == "RECHAZA":
        return "bad"
    return "review"


def curate_images(rows: list) -> list:
    """Copia imágenes a las carpetas curadas. Retorna lista de filas enriquecidas."""
    for g in GROUPS:
        (CURATED_ROOT / g).mkdir(parents=True, exist_ok=True)

    curated = []
    for row in rows:
        img_name = row["image"]
        src = SRC_IMGS / img_name
        group = assign_group(row)
        dst = CURATED_ROOT / group / img_name

        copied = False
        if src.exists():
            shutil.copy2(src, dst)
            copied = True
        else:
            print(f"  AVISO: imagen no encontrada: {src}")

        curated.append({
            "image":              img_name,
            "source_path":        str(src),
            "curated_group":      group,
            "original_decision":  row.get("decision", ""),
            "display_label":      row.get("display_label", ""),
            "defect_pct":         row.get("defect_pct", ""),
            "dark_rot_pct":       row.get("dark_rot_pct", ""),
            "max_region_pct":     row.get("max_region_pct", ""),
            "brown_dark_pct":     row.get("brown_dark_pct", ""),
            "quality_cls_pred":   row.get("quality_cls_pred", ""),
            "quality_cls_good_conf": row.get("quality_cls_good_conf", ""),
            "quality_cls_bad_conf":  row.get("quality_cls_bad_conf", ""),
            "yolo_defect_valid":  row.get("yolo_defect_count", ""),
            "mask_quality_ok":    row.get("mask_quality_ok", ""),
            "mask_fail_reason":   row.get("mask_fail_reason", ""),
            "_copied":            copied,
        })

    return curated


# ---------------------------------------------------------------------------
# TAREA 3: CSV curado
# ---------------------------------------------------------------------------

def write_curated_csv(curated: list):
    path = CURATED_ROOT / "curated_labels.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=EXPORT_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(curated)
    print(f"  CSV curado: {path} ({len(curated)} filas)")
    return path


# ---------------------------------------------------------------------------
# TAREA 4: Dataset de clasificación
# ---------------------------------------------------------------------------

def make_cls_dataset(curated: list) -> dict:
    """Split estratificado 70/20/10 sobre good y bad. Retorna conteos."""
    rng = random.Random(SEED)

    by_class = {"good": [], "bad": []}
    for row in curated:
        g = row["curated_group"]
        if g in by_class and row["_copied"]:
            by_class[g].append(row["image"])

    counts = {sp: {cls: 0 for cls in ["good", "bad"]} for sp in ["train", "val", "test"]}

    for cls, imgs in by_class.items():
        rng_local = random.Random(SEED)
        shuffled = list(imgs)
        rng_local.shuffle(shuffled)
        n = len(shuffled)
        n_train = math.floor(n * SPLIT["train"])
        n_val   = math.floor(n * SPLIT["val"])
        splits_map = {
            "train": shuffled[:n_train],
            "val":   shuffled[n_train:n_train + n_val],
            "test":  shuffled[n_train + n_val:],
        }
        for sp, sp_imgs in splits_map.items():
            dst_dir = CLS_ROOT / sp / cls
            dst_dir.mkdir(parents=True, exist_ok=True)
            for img_name in sp_imgs:
                src = CURATED_ROOT / cls / img_name
                if src.exists():
                    shutil.copy2(src, dst_dir / img_name)
                    counts[sp][cls] += 1

    return counts


# ---------------------------------------------------------------------------
# TAREA 5: Previews
# ---------------------------------------------------------------------------

GRID_COLORS = {
    "good":                     (0,  200,  80),
    "bad":                      (220,  50,  50),
    "review":                   (220, 180,   0),
    "excluded_invalid_capture": (120, 120, 120),
}

GRID_BG = {
    "good": (22, 40, 22), "bad": (40, 22, 22),
    "review": (40, 38, 22), "excluded_invalid_capture": (30, 30, 30),
}


def _make_grid(title: str, entries: list, out_path: Path,
               cols: int = 10, thumb_w: int = 140, thumb_h: int = 130,
               label_h: int = 22, header_h: int = 32):
    if not entries:
        img = Image.new("RGB", (400, 80), (30, 30, 30))
        ImageDraw.Draw(img).text((10, 10), f"{title} — sin imágenes", fill=(180, 180, 180))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(out_path), quality=88)
        return

    cell_h = thumb_h + label_h
    rows_n = max(1, math.ceil(len(entries) / cols))
    sheet_w = thumb_w * cols
    sheet_h = header_h + cell_h * rows_n

    sheet = Image.new("RGB", (sheet_w, sheet_h), (22, 22, 22))
    draw  = ImageDraw.Draw(sheet)
    f_hdr = _get_font(14)
    f_sm  = _get_font(10)

    draw.text((6, 6), title, fill=(220, 220, 220), font=f_hdr)

    for i, entry in enumerate(entries):
        col = i % cols
        row = i // cols
        x0  = col * thumb_w
        y0  = header_h + row * cell_h

        grp   = entry["group"]
        color = GRID_COLORS.get(grp, (180, 180, 180))
        bg    = GRID_BG.get(grp, (30, 30, 30))
        draw.rectangle([(x0, y0), (x0 + thumb_w, y0 + cell_h)], fill=bg)

        thumb, ok = _load_thumb(entry["path"], thumb_w - 2, thumb_h - 2)
        sheet.paste(thumb, (x0 + 1, y0 + 1))

        label = entry.get("label", grp)[:18]
        draw.rectangle([(x0, y0 + thumb_h), (x0 + thumb_w, y0 + cell_h)], fill=(30, 30, 30))
        draw.text((x0 + 2, y0 + thumb_h + 2), label, fill=color, font=f_sm)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(str(out_path), quality=88)
    print(f"  Grid: {out_path}")


def make_previews(curated: list):
    by_group = {g: [] for g in GROUPS}
    for row in curated:
        g = row["curated_group"]
        src = CURATED_ROOT / g / row["image"]
        label = f"def={row.get('defect_pct','?')}%"
        by_group[g].append({"path": src, "group": g, "label": label})

    for g in GROUPS:
        _make_grid(
            title=f"Fruits-360 Curated — {g.upper()} ({len(by_group[g])} imgs)",
            entries=by_group[g][:60],
            out_path=PREVIEW_DIR / f"{g}_grid.jpg",
        )

    # Grid mixto: hasta 20 de cada grupo, mezclados con etiqueta
    mixed = []
    rng = random.Random(SEED)
    for g in GROUPS:
        sample = list(by_group[g])
        rng.shuffle(sample)
        mixed.extend(sample[:20])
    rng.shuffle(mixed)
    _make_grid(
        title=f"Fruits-360 Curated — MIXED ({len(mixed)} muestra)",
        entries=mixed,
        out_path=PREVIEW_DIR / "mixed_curated_grid.jpg",
    )


# ---------------------------------------------------------------------------
# TAREA 6: Reportes
# ---------------------------------------------------------------------------

def _count_by_group(curated):
    counts = {g: 0 for g in GROUPS}
    for row in curated:
        counts[row["curated_group"]] = counts.get(row["curated_group"], 0) + 1
    return counts


def write_report(curated: list, cls_counts: dict):
    group_counts = _count_by_group(curated)
    total = len(curated)

    lines = [
        "# Fruits-360 Quality Curation v1 — Informe",
        "",
        f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Origen",
        f"- Dataset evaluado: `data/samples_quality_fruits360_good_eval/`",
        f"- CSV de resultados: `outputs/quality_analysis_fruits360_good_eval/resultados_calidad.csv`",
        f"- Total imágenes procesadas: {total}",
        "",
        "## Distribución del curado",
        "",
        "| Grupo | Cantidad | % |",
        "|-------|----------|---|",
    ]
    for g in GROUPS:
        n = group_counts.get(g, 0)
        pct = n / max(1, total) * 100
        lines.append(f"| {g} | {n} | {pct:.1f}% |")

    lines += [
        "",
        "## Dataset de clasificación (good / bad)",
        "",
        "| Split | good | bad | total |",
        "|-------|------|-----|-------|",
    ]
    for sp in ["train", "val", "test"]:
        g = cls_counts[sp]["good"]
        b = cls_counts[sp]["bad"]
        lines.append(f"| {sp} | {g} | {b} | {g+b} |")

    total_cls_g = sum(cls_counts[sp]["good"] for sp in ["train","val","test"])
    total_cls_b = sum(cls_counts[sp]["bad"]  for sp in ["train","val","test"])
    lines += [
        f"| **TOTAL** | {total_cls_g} | {total_cls_b} | {total_cls_g+total_cls_b} |",
        "",
        "## Criterios de curado (automático, basados en pipeline)",
        "",
        "- **excluded_invalid_capture**: `capture_valid=False`, máscara inválida, clasificador sin crop, o todas las métricas a cero.",
        "- **good**: `decision=PASA` con captura válida.",
        "- **bad**: `decision=RECHAZA` con captura válida.",
        "- **review**: `decision=REVISAR` con captura válida.",
        "",
        "## Advertencia",
        "",
        "> El curado es **automático** basado en las métricas del pipeline rule-based.",
        "> No ha sido revisado imagen por imagen por un humano.",
        "> Úsalo como dataset auxiliar para entrenamiento, no como verdad absoluta.",
        "> Antes de entrenar, se recomienda revisar visualmente las muestras `bad` y `review`.",
        "",
        "## Recomendación",
        "",
        "- Usar `good` y `bad` para entrenamiento del clasificador de calidad.",
        "- No usar `review` en entrenamiento (etiqueta ambigua).",
        "- No usar `excluded_invalid_capture` en entrenamiento.",
        "- Complementar con las peras propias ya etiquetadas en `data/samples_quality_controlled_test`.",
        "",
        "## Rutas finales",
        "",
        f"- Curado: `data/quality_curated_fruits360_v1/`",
        f"- Clasificación: `data/quality_curated_fruits360_cls_v1/`",
        f"- Previews: `outputs/fruits360_curated_quality_preview/`",
    ]

    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Reporte: {REPORT_MD}")


def write_validation(curated: list, cls_counts: dict):
    group_counts = _count_by_group(curated)
    issues = []
    ok_checks = []

    # 1. Imágenes copiadas correctamente
    missing = [r["image"] for r in curated if not r["_copied"]]
    if missing:
        issues.append(f"  FALLO: {len(missing)} imágenes no encontradas en origen: {missing[:5]}")
    else:
        ok_checks.append(f"  OK  Todas las {len(curated)} imágenes copiadas correctamente.")

    # 2. No hay imágenes corruptas (verificar que OpenCV puede leerlas)
    corrupt = []
    for g in GROUPS:
        for f in (CURATED_ROOT / g).iterdir():
            if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                try:
                    buf = np.frombuffer(f.read_bytes(), dtype=np.uint8)
                    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                    if img is None:
                        corrupt.append(str(f))
                except Exception:
                    corrupt.append(str(f))
    if corrupt:
        issues.append(f"  FALLO: {len(corrupt)} imágenes corruptas: {corrupt[:3]}")
    else:
        ok_checks.append(f"  OK  Sin imágenes corruptas en las 4 carpetas curadas.")

    # 3. No hay solapamiento entre train/val/test
    sets_by_split = {}
    for sp in ["train", "val", "test"]:
        imgs = set()
        for cls in ["good", "bad"]:
            d = CLS_ROOT / sp / cls
            if d.exists():
                imgs.update(f.name for f in d.iterdir())
        sets_by_split[sp] = imgs

    overlaps = []
    sp_list = list(sets_by_split.keys())
    for i in range(len(sp_list)):
        for j in range(i+1, len(sp_list)):
            inter = sets_by_split[sp_list[i]] & sets_by_split[sp_list[j]]
            if inter:
                overlaps.append(f"{sp_list[i]}&{sp_list[j]}: {list(inter)[:3]}")
    if overlaps:
        issues.append(f"  FALLO: solapamiento entre splits: {overlaps}")
    else:
        ok_checks.append("  OK  Sin solapamiento entre train/val/test.")

    # 4. No hay imágenes de review en cls
    review_in_cls = []
    review_names = {r["image"] for r in curated if r["curated_group"] == "review"}
    for sp in ["train", "val", "test"]:
        for cls in ["good", "bad"]:
            d = CLS_ROOT / sp / cls
            if d.exists():
                found = [f.name for f in d.iterdir() if f.name in review_names]
                review_in_cls.extend(found)
    if review_in_cls:
        issues.append(f"  FALLO: {len(review_in_cls)} imágenes de review en CLS dataset.")
    else:
        ok_checks.append("  OK  Ninguna imagen de 'review' en el dataset de clasificación.")

    # 5. No hay imágenes de excluded_invalid_capture en cls
    excluded_names = {r["image"] for r in curated if r["curated_group"] == "excluded_invalid_capture"}
    excl_in_cls = []
    for sp in ["train", "val", "test"]:
        for cls in ["good", "bad"]:
            d = CLS_ROOT / sp / cls
            if d.exists():
                found = [f.name for f in d.iterdir() if f.name in excluded_names]
                excl_in_cls.extend(found)
    if excl_in_cls:
        issues.append(f"  FALLO: {len(excl_in_cls)} imágenes excluidas en CLS dataset.")
    else:
        ok_checks.append("  OK  Ninguna imagen excluida en el dataset de clasificación.")

    # 6. Conteos cuadran
    total_cls = sum(cls_counts[sp][c] for sp in ["train","val","test"] for c in ["good","bad"])
    expected  = group_counts.get("good", 0) + group_counts.get("bad", 0)
    if total_cls != expected:
        issues.append(f"  AVISO: total cls={total_cls} vs good+bad en curated={expected} (diferencia por redondeo de split puede ser normal).")
    else:
        ok_checks.append(f"  OK  Conteos cuadran: {total_cls} imágenes en CLS = good+bad curados.")

    status = "PASS" if not issues else "FAIL"
    lines = [
        "=== Fruits-360 Quality Curation v1 — Validation ===",
        f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Status: {status}",
        "",
        "Checks OK:",
    ] + ok_checks

    if issues:
        lines += ["", "Checks FAIL:"] + issues

    lines += [
        "",
        "=== Resumen de conteos ===",
        f"  Total curado: {len(curated)}",
    ]
    for g in GROUPS:
        lines.append(f"  {g}: {group_counts.get(g,0)}")
    lines.append("")
    for sp in ["train", "val", "test"]:
        lines.append(f"  CLS {sp}: good={cls_counts[sp]['good']} bad={cls_counts[sp]['bad']}")
    lines.append("=== Fin ===")

    VALIDATION_TXT.parent.mkdir(parents=True, exist_ok=True)
    VALIDATION_TXT.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Validacion: {VALIDATION_TXT}")
    return status, group_counts


# ---------------------------------------------------------------------------
# TAREA 7: main
# ---------------------------------------------------------------------------

def main():
    print("=== curate_fruits360_quality_dataset ===")

    if not RESULTS_CSV.exists():
        print(f"ERROR: {RESULTS_CSV} no existe.")
        sys.exit(1)
    if not SRC_IMGS.exists():
        print(f"ERROR: {SRC_IMGS} no existe.")
        sys.exit(1)

    with open(RESULTS_CSV, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"  CSV leido: {len(rows)} filas")

    # TAREA 1+2: copiar imágenes curadas
    print("  Curando y copiando imágenes...")
    curated = curate_images(rows)

    group_counts = _count_by_group(curated)
    print(f"  good={group_counts['good']}  bad={group_counts['bad']}  "
          f"review={group_counts['review']}  excluded={group_counts['excluded_invalid_capture']}")

    # TAREA 3: CSV curado
    write_curated_csv(curated)

    # TAREA 4: dataset clasificación
    print("  Creando dataset de clasificación...")
    cls_counts = make_cls_dataset(curated)
    for sp in ["train", "val", "test"]:
        print(f"    {sp}: good={cls_counts[sp]['good']}  bad={cls_counts[sp]['bad']}")

    # TAREA 5: previews
    print("  Generando grids de preview...")
    make_previews(curated)

    # TAREA 6: reportes
    write_report(curated, cls_counts)
    status, group_counts = write_validation(curated, cls_counts)

    # Salida final
    total_cls_g = sum(cls_counts[sp]["good"] for sp in ["train","val","test"])
    total_cls_b = sum(cls_counts[sp]["bad"]  for sp in ["train","val","test"])

    print()
    print("FRUITS360 QUALITY CURATION COMPLETADA")
    print()
    print(f"TOTAL_PROCESSED: {len(curated)}")
    print(f"GOOD: {group_counts['good']}")
    print(f"BAD: {group_counts['bad']}")
    print(f"REVIEW: {group_counts['review']}")
    print(f"EXCLUDED_INVALID_CAPTURE: {group_counts['excluded_invalid_capture']}")
    print()
    print(f"CLS_DATASET:")
    print(f"TRAIN_GOOD: {cls_counts['train']['good']}")
    print(f"TRAIN_BAD:  {cls_counts['train']['bad']}")
    print(f"VAL_GOOD:   {cls_counts['val']['good']}")
    print(f"VAL_BAD:    {cls_counts['val']['bad']}")
    print(f"TEST_GOOD:  {cls_counts['test']['good']}")
    print(f"TEST_BAD:   {cls_counts['test']['bad']}")
    print()
    print("RUTAS:")
    print(f"CURATED_ROOT: {CURATED_ROOT}")
    print(f"CLS_ROOT:     {CLS_ROOT}")
    print(f"PREVIEW:      {PREVIEW_DIR}")
    print(f"REPORT:       {REPORT_MD}")
    print(f"VALIDATION:   {VALIDATION_TXT}")
    print()
    print(f"STATUS: {status}")


if __name__ == "__main__":
    main()
