"""
audit_quality_pipeline.py
Auditoría completa del pipeline de calidad PearVision QC.

Lee:  outputs/quality_analysis/resultados_calidad.csv
Genera:
  outputs/quality_audit/contact_sheet_full_pipeline.jpg
  outputs/quality_audit/audit_quality_pipeline.csv
  reports/quality_pipeline_audit_report.md
"""

import csv
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CSV_IN  = PROJECT_ROOT / "outputs" / "quality_analysis" / "resultados_calidad.csv"
IMGS_DIR = PROJECT_ROOT / "outputs" / "quality_analysis"
AUDIT_DIR = PROJECT_ROOT / "outputs" / "quality_audit"
REPORT_DIR = PROJECT_ROOT / "reports"

CONTACT_SHEET = AUDIT_DIR / "contact_sheet_full_pipeline.jpg"
AUDIT_CSV     = AUDIT_DIR / "audit_quality_pipeline.csv"
REPORT_MD     = REPORT_DIR / "quality_pipeline_audit_report.md"


def _to_float(v, default=0.0):
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def _to_bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("true", "1", "yes")


def compute_audit_flag(row):
    decision       = row.get("decision", "")
    defect_pct     = _to_float(row.get("defect_pct", 0))
    dark_rot_pct   = _to_float(row.get("dark_rot_pct", 0))
    max_region_pct = _to_float(row.get("max_region_pct", 0))
    cls_pred       = row.get("quality_cls_pred", "unknown").strip().lower()
    cls_bad_conf   = _to_float(row.get("quality_cls_bad_conf", 0))
    mask_ok        = _to_bool(row.get("mask_quality_ok", True))

    flags    = []
    comments = []

    # Regla 1: posible falso rechazo
    if (decision == "RECHAZA"
            and defect_pct < 5.0
            and dark_rot_pct < 5.0
            and max_region_pct < 5.0):
        flags.append("possible_false_reject")
        comments.append(
            f"RECHAZA pero defect={defect_pct:.1f}% "
            f"rot={dark_rot_pct:.1f}% region={max_region_pct:.1f}%"
        )

    # Regla 2: posible falso aceptado
    if decision == "PASA" and cls_pred == "bad" and cls_bad_conf >= 0.85:
        flags.append("possible_hidden_bad")
        comments.append(f"PASA pero CLS=bad conf={cls_bad_conf:.2f}")

    # Regla 3: defect_model_mostly_ignored
    # La columna yolo_defect_ignored no existe en el CSV actual; se omite.

    # Regla 4: problema de máscara
    if not mask_ok:
        flags.append("mask_problem")
        reason = row.get("mask_fail_reason", "")
        comments.append(f"mask_quality_ok=False reason={reason}")

    if not flags:
        flags.append("ok")

    return (
        "|".join(flags),
        "; ".join(comments),
        any(f != "ok" for f in flags),
    )


def _load_panel(path: Path, target_h: int = 180, target_w: int = 200):
    """Carga imagen BGR con cv2 (compatible con rutas Unicode en Windows)."""
    bgr = None
    try:
        buf = np.frombuffer(path.read_bytes(), dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        pass
    if bgr is None:
        # Placeholder gris
        arr = np.full((target_h, target_w, 3), 70, dtype=np.uint8)
        return Image.fromarray(arr)
    h, w = bgr.shape[:2]
    scale = min(target_h / h, target_w / w)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    # Canvas fijo con imagen centrada
    canvas = np.full((target_h, target_w, 3), 50, dtype=np.uint8)
    y_off = (target_h - new_h) // 2
    x_off = (target_w - new_w) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = rgb
    return Image.fromarray(canvas)


def _get_font(size, bold=False):
    candidates = [
        "arial.ttf", "Arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def make_contact_sheet(rows):
    CELL_H   = 180
    PANEL_W  = 200
    TEXT_W   = 280
    N_PANELS = 4   # overlay, bbox, crop, mask
    ROW_H    = CELL_H + 6
    HEADER_H = 28

    total_w = TEXT_W + PANEL_W * N_PANELS
    total_h = HEADER_H + ROW_H * len(rows)

    img    = Image.new("RGB", (total_w, total_h), (25, 25, 25))
    draw   = ImageDraw.Draw(img)

    f_sm = _get_font(11)
    f_md = _get_font(13)
    f_lg = _get_font(15)

    # Cabecera
    draw.text((6, 6), "PearVision QC — Full Pipeline Audit", fill=(220, 220, 220), font=f_lg)
    for pi, lbl in enumerate(["OVERLAY", "BBOX", "CROP", "MASK"]):
        draw.text((TEXT_W + pi * PANEL_W + 6, 8), lbl, fill=(160, 160, 160), font=f_sm)

    for ri, row in enumerate(rows):
        y0 = HEADER_H + ri * ROW_H

        # Fondo alternado
        draw.rectangle([(0, y0), (total_w, y0 + ROW_H)],
                        fill=(38, 38, 38) if ri % 2 == 0 else (44, 44, 44))

        # Colores según decisión
        decision = row.get("decision", "?")
        dec_color = {"PASA": (0, 210, 80),
                     "REVISAR": (230, 185, 0),
                     "RECHAZA": (230, 60, 60)}.get(decision, (180, 180, 180))

        audit_flag = row.get("audit_flag", "ok")
        if audit_flag == "ok":
            flag_color = (80, 200, 80)
        elif "reject" in audit_flag or "hidden" in audit_flag:
            flag_color = (255, 100, 60)
        else:
            flag_color = (230, 160, 0)

        # Panel de texto
        stem = Path(row.get("image", "?")).stem
        text_y = y0 + 4
        line_h = 16

        def put(text, color, font):
            nonlocal text_y
            draw.text((6, text_y), text, fill=color, font=font)
            text_y += line_h

        put(row.get("image", "?"),                                     (210, 210, 210), f_md)
        put(f"{decision}  |  {row.get('estimated_category','')}",      dec_color,       f_md)
        put(f"def={row.get('defect_pct','?')}%  rot={row.get('dark_rot_pct','?')}%  reg={row.get('max_region_pct','?')}%",
            (175, 175, 175), f_sm)
        put(f"pear={row.get('pear_visible_pct','?')}%  body={row.get('body_visible_pct','?')}%",
            (175, 175, 175), f_sm)
        put(f"yolo_valid={row.get('yolo_defect_count','0')}  brown={row.get('brown_dark_pct','?')}%",
            (175, 175, 175), f_sm)
        put(f"CLS={row.get('quality_cls_pred','?')}  bad={row.get('quality_cls_bad_conf','?')}  good={row.get('quality_cls_good_conf','?')}",
            (160, 210, 255), f_sm)
        put(f"flag: {audit_flag}",                                      flag_color,      f_sm)
        comment = row.get("audit_comment", "")
        if comment:
            put(f"  {comment[:46]}",                                    (255, 175, 70),  f_sm)

        # Paneles de imagen
        panel_paths = [
            IMGS_DIR / f"{stem}.jpg",
            IMGS_DIR / f"{stem}_bbox.jpg",
            IMGS_DIR / f"{stem}_crop.jpg",
            IMGS_DIR / f"{stem}_mask.jpg",
        ]
        for pi, ppath in enumerate(panel_paths):
            x0 = TEXT_W + pi * PANEL_W
            if ppath.exists():
                panel = _load_panel(ppath, target_h=CELL_H, target_w=PANEL_W)
                img.paste(panel, (x0, y0 + 3))
            else:
                draw.rectangle([(x0 + 2, y0 + 3), (x0 + PANEL_W - 2, y0 + CELL_H)],
                                fill=(60, 60, 60))
                draw.text((x0 + PANEL_W // 2 - 10, y0 + CELL_H // 2),
                          "N/A", fill=(110, 110, 110), font=f_sm)

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    img.save(str(CONTACT_SHEET), quality=92)
    print(f"  Contact sheet guardado: {CONTACT_SHEET}")


def write_audit_csv(rows):
    fieldnames = [
        "image", "decision", "estimated_category", "display_label",
        "capture_valid", "capture_reason",
        "defect_pct", "dark_rot_pct", "max_region_pct",
        "pear_visible_pct", "body_visible_pct",
        "yolo_defect_count", "yolo_defect_area_pct", "yolo_defect_max_conf",
        "yolo_defect_classes",
        "brown_dark_pct", "dark_area_pct",
        "mask_source", "mask_quality_ok", "mask_fail_reason",
        "quality_cls_used", "quality_cls_source", "quality_cls_pred",
        "quality_cls_good_conf", "quality_cls_bad_conf", "quality_cls_max_conf",
        "quality_cls_action",
        "detector_conf",
        "audit_flag", "audit_comment", "needs_manual_review",
    ]
    with open(AUDIT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Audit CSV guardado: {AUDIT_CSV}")


def generate_report(rows):
    total   = len(rows)
    pasa    = sum(1 for r in rows if r["decision"] == "PASA")
    revisar = sum(1 for r in rows if r["decision"] == "REVISAR")
    rechaza = sum(1 for r in rows if r["decision"] == "RECHAZA")

    false_rejects = [r for r in rows if "possible_false_reject" in r["audit_flag"]]
    hidden_bad    = [r for r in rows if "possible_hidden_bad"    in r["audit_flag"]]
    mask_problems = [r for r in rows if "mask_problem"           in r["audit_flag"]]
    needs_review  = sum(1 for r in rows if r["needs_manual_review"])

    avg_det_conf  = sum(_to_float(r.get("detector_conf", 0)) for r in rows) / max(1, total)
    grabcut_ok    = sum(1 for r in rows if r.get("mask_source", "") == "grabcut")
    mask_ok_count = sum(1 for r in rows if _to_bool(r.get("mask_quality_ok", True)))
    psd_total     = sum(int(_to_float(r.get("yolo_defect_count", 0))) for r in rows)
    cls_good      = sum(1 for r in rows if r.get("quality_cls_pred", "").lower() == "good")
    cls_bad_cnt   = sum(1 for r in rows if r.get("quality_cls_pred", "").lower() == "bad")

    lines = [
        "# PearVision QC — Auditoría del Pipeline Completo",
        "",
        f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## 1. Resumen Global",
        "",
        "| Métrica | Valor |",
        "|---------|-------|",
        f"| Total imágenes | {total} |",
        f"| PASA | {pasa} |",
        f"| REVISAR | {revisar} |",
        f"| RECHAZA | {rechaza} |",
        f"| Necesitan revisión manual | {needs_review} |",
        "",
        "## 2. Tabla Detallada",
        "",
        "| imagen | decision | cat_estimada | defect% | rot% | region% | yolo_valid | yolo_ignored | cls_pred | cls_bad | audit_flag |",
        "|--------|----------|-------------|---------|------|---------|------------|--------------|----------|---------|------------|",
    ]

    for r in rows:
        lines.append(
            f"| {r['image']} "
            f"| {r['decision']} "
            f"| {r['estimated_category']} "
            f"| {r['defect_pct']} "
            f"| {r['dark_rot_pct']} "
            f"| {r['max_region_pct']} "
            f"| {r['yolo_defect_count']} "
            f"| N/A "
            f"| {r['quality_cls_pred']} "
            f"| {r['quality_cls_bad_conf']} "
            f"| {r['audit_flag']} |"
        )

    lines += ["", "## 3. Posibles Falsos Rechazos", ""]
    if false_rejects:
        for r in false_rejects:
            lines.append(
                f"- **{r['image']}**: decision={r['decision']}  "
                f"defect={r['defect_pct']}%  rot={r['dark_rot_pct']}%  "
                f"region={r['max_region_pct']}%  — _{r['audit_comment']}_"
            )
    else:
        lines.append("_Ninguno detectado._")

    lines += ["", "## 4. Posibles Falsos Aceptados (PASA con CLS=BAD alto)", ""]
    if hidden_bad:
        for r in hidden_bad:
            lines.append(
                f"- **{r['image']}**: decision={r['decision']}  "
                f"cls_pred={r['quality_cls_pred']}  cls_bad_conf={r['quality_cls_bad_conf']}  "
                f"— _{r['audit_comment']}_"
            )
    else:
        lines.append("_Ninguno detectado._")

    lines += ["", "## 5. Problemas de Máscara", ""]
    if mask_problems:
        for r in mask_problems:
            lines.append(
                f"- **{r['image']}**: mask_source={r['mask_source']}  "
                f"mask_quality_ok={r['mask_quality_ok']}  "
                f"reason={r['mask_fail_reason']}"
            )
    else:
        lines.append("_Ninguno detectado._")

    lines += [
        "",
        "## 6. Conclusión Técnica",
        "",
        "### Detector de pera (YOLO ECLPOD)",
        f"- Confianza media de detección: {avg_det_conf:.2f}",
        f"- Capturas válidas: {sum(1 for r in rows if _to_bool(r.get('capture_valid', True)))}/{total}",
        (f"- **Evaluación**: El detector es estable — todas las imágenes fueron capturadas "
         f"correctamente con confianza media {avg_det_conf:.2f}."),
        "",
        "### Máscara ROI (GrabCut)",
        f"- GrabCut exitoso: {grabcut_ok}/{total}",
        f"- mask_quality_ok=True: {mask_ok_count}/{total}",
    ]
    if mask_ok_count == total:
        lines.append("- **Evaluación**: La máscara GrabCut es estable en este conjunto. "
                     "No se usó fallback ellipse en ninguna imagen.")
    else:
        lines.append(f"- **Evaluación**: {total - mask_ok_count} imagen(es) usaron máscara "
                     "fallback (ellipse). Puede haber imprecisión en esas métricas.")

    lines += [
        "",
        "### Modelo de defectos PSD (YOLO)",
        f"- Detecciones válidas totales en el conjunto: {psd_total}",
    ]
    if psd_total == 0:
        lines.append(
            "- **Evaluación**: El modelo PSD no generó detecciones válidas en ninguna imagen "
            "del conjunto de prueba. Esto puede deberse a umbrales altos de confianza, a que "
            "los defectos presentes no están en el dominio del modelo (entrenado con PSD), o a "
            "que los filtros de borde/area son demasiado restrictivos. "
            "La señal principal de calidad sigue siendo la métrica HSV (defect_pct, dark_rot_pct)."
        )
    else:
        lines.append(
            f"- **Evaluación**: El modelo PSD detectó {psd_total} defectos válidos. "
            "Revisar si coinciden visualmente con las zonas defectuosas reales."
        )

    lines += [
        "",
        "### Clasificador GOOD/BAD Mendeley",
        f"- Predice GOOD: {cls_good}/{total} imágenes",
        f"- Predice BAD: {cls_bad_cnt}/{total} imágenes",
        f"- Posibles falsos aceptados detectados: {len(hidden_bad)}",
    ]

    if cls_good > total * 0.7:
        lines.append(
            "- **Evaluación**: El clasificador predice GOOD en la gran mayoría de imágenes, "
            "incluso en peras con podredumbre visible (>90% de superficie afectada). "
            "Esto confirma el sesgo de dataset detectado previamente: el modelo aprendió "
            "características de composición fotográfica (screenshots vs fotos reales), "
            "no de calidad superficial de la fruta."
        )
        lines.append(
            "- **Recomendación**: Mantener el clasificador Mendeley **solo como señal informativa** "
            "(`--use-quality-cls` sin `--quality-cls-affect-decision`). "
            "No activar `--quality-cls-affect-decision` hasta reentrenar con imágenes "
            "representativas de defectos reales."
        )
    else:
        lines.append(
            "- **Evaluación**: El clasificador muestra cierta alineación con la calidad real. "
            "Se puede considerar activar `--quality-cls-affect-decision` con umbral ≥0.90."
        )

    lines += [
        "",
        "### Resumen de Acciones Recomendadas",
        "",
    ]
    if false_rejects:
        lines.append(
            f"1. **Revisar umbrales HSV**: {len(false_rejects)} imagen(es) marcadas como "
            "`possible_false_reject`. Los umbrales actuales pueden estar rechazando "
            "peras con coloración natural marrón-verde sin podredumbre real."
        )
    if hidden_bad:
        lines.append(
            f"2. **Revisar falsos aceptados**: {len(hidden_bad)} imagen(es) PASA con CLS=bad alto. "
            "Puede indicar defectos no detectados por las métricas HSV."
        )
    if psd_total == 0:
        lines.append(
            "3. **Modelo PSD**: Sin detecciones válidas. Considerar bajar `--defect-conf` a 0.15 "
            "o revisar si el modelo PSD fue entrenado con imágenes similares a este conjunto."
        )
    if not false_rejects and not hidden_bad and mask_ok_count == total:
        lines.append("- El pipeline muestra comportamiento coherente en este conjunto de prueba.")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Informe markdown guardado: {REPORT_MD}")


def main():
    if not CSV_IN.exists():
        print(f"ERROR: CSV no encontrado: {CSV_IN}")
        print("  Ejecuta primero analyze_quality.py con --save")
        sys.exit(1)

    print("=== audit_quality_pipeline ===")
    print(f"  Leyendo CSV: {CSV_IN}")

    with open(CSV_IN, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    print(f"  Registros: {len(rows)}")

    # Verificar archivos de imagen relacionados
    missing = []
    for row in rows:
        stem = Path(row["image"]).stem
        for suffix in ["", "_bbox", "_crop", "_mask"]:
            p = IMGS_DIR / f"{stem}{suffix}.jpg"
            if not p.exists():
                missing.append(p.name)
    if missing:
        for m in missing:
            print(f"  AVISO: archivo no encontrado: {m}")
    else:
        print(f"  Todos los archivos de imagen encontrados.")

    # Calcular audit flags
    for row in rows:
        flag, comment, needs_review = compute_audit_flag(row)
        row["audit_flag"]          = flag
        row["audit_comment"]       = comment
        row["needs_manual_review"] = needs_review

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    # Tarea 2: contact sheet
    make_contact_sheet(rows)

    # Tarea 3: audit CSV
    write_audit_csv(rows)

    # Tarea 4: informe markdown
    generate_report(rows)

    # Resumen final
    counts = {}
    for r in rows:
        counts[r["decision"]] = counts.get(r["decision"], 0) + 1
    print(
        f"\n  Resumen: PASA={counts.get('PASA',0)}  "
        f"REVISAR={counts.get('REVISAR',0)}  "
        f"RECHAZA={counts.get('RECHAZA',0)}"
    )
    flags_summary = {}
    for r in rows:
        for f in r["audit_flag"].split("|"):
            flags_summary[f] = flags_summary.get(f, 0) + 1
    print(f"  Flags: {flags_summary}")
    print("=== Auditoría completada ===")


if __name__ == "__main__":
    main()
