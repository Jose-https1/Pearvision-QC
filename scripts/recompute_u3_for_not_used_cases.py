"""
recompute_u3_for_not_used_cases.py

TAREA 2: Ejecuta U3 directamente sobre las 6 imagenes que quedaron en REVISAR
         porque capture_valid=False bloqueo U3 en el pipeline original.

TAREA 4: Genera la evaluacion fusion fixed v2 combinando los 80 PASA ya
         correctos con las predicciones recalculadas de las 6 imagenes.

No entrena ningun modelo. No modifica V2, U3 ni quality_rules.yaml.
"""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

SOURCE_DIRS = [
    PROJECT_ROOT / "data/unseen_quality_eval_input/supermarket_valid_conditions_batch_v3",
    PROJECT_ROOT / "data/unseen_quality_eval_input/supermarket_good_batch_v2",
    PROJECT_ROOT / "data/supermarket_good_hard_examples_v1/images",
    PROJECT_ROOT / "data/supermarket_good_hard_examples_v2/images",
]

NOT_USED_IMAGES = [
    "1000060736.jpg",
    "1000060738.jpg",
    "1000060739.jpg",
    "1000060740.jpg",
    "1000060741.jpg",
    "1000060745.jpg",
]

U3_MODEL_PATH = PROJECT_ROOT / "outputs/fruits360_quality_cls_u3_roi_masked_clean/best_model.pt"
U3_THR_PATH   = PROJECT_ROOT / "outputs/fruits360_quality_cls_u3_roi_masked_clean/selected_thresholds.json"
INPUT_CSV     = PROJECT_ROOT / "outputs/u3_fusion_fixed_eval/resultados_u3_fusion_fixed.csv"

NOT_USED_EVAL_DIR = PROJECT_ROOT / "outputs/u3_not_used_cases_eval"
V2_DIR            = PROJECT_ROOT / "outputs/u3_fusion_fixed_eval_v2"
REPORTS_DIR       = PROJECT_ROOT / "reports"

NOT_USED_EVAL_DIR.mkdir(parents=True, exist_ok=True)
V2_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DECISION_COLOR = {"PASA": (0, 200, 0), "REVISAR": (0, 165, 255), "RECHAZA": (0, 0, 220)}
THUMB_W, THUMB_H = 280, 280
COLS = 5


# ---------------------------------------------------------------------------
# Helpers de imagen
# ---------------------------------------------------------------------------
def find_image(name: str) -> Path:
    for src in SOURCE_DIRS:
        c = src / name
        if c.exists():
            return c
    return None


def load_thumb(img_path, w=THUMB_W, h=THUMB_H):
    if img_path is None or not Path(str(img_path)).exists():
        return np.full((h, w, 3), 80, dtype=np.uint8)
    try:
        from PIL import Image
        pil = Image.open(str(img_path)).convert("RGB")
        pil = pil.resize((w, h))
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        return np.full((h, w, 3), 80, dtype=np.uint8)


def add_overlay(thumb, image_name: str, old_dec: str, new_dec: str,
                u3_raw: str, p_good: float, p_bad: float, reason: str):
    color = DECISION_COLOR.get(new_dec, (200, 200, 200))
    cv2.rectangle(thumb, (0, 0), (thumb.shape[1]-1, thumb.shape[0]-1), color, 5)
    lines = [
        image_name[:22],
        f"ANT:{old_dec[:7]}  NOW:{new_dec}",
        f"U3:{u3_raw}  g={p_good:.3f} b={p_bad:.3f}",
        reason[:34],
    ]
    overlay_h = len(lines) * 18 + 8
    h, w = thumb.shape[:2]
    cv2.rectangle(thumb, (0, h - overlay_h), (w, h), (15, 15, 15), -1)
    for i, line in enumerate(lines):
        y = h - overlay_h + 15 + i * 17
        cv2.putText(thumb, line, (3, y), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(thumb, line, (3, y), cv2.FONT_HERSHEY_SIMPLEX, 0.37, color, 1, cv2.LINE_AA)
    return thumb


def add_overlay_from_row(thumb, row: dict, new_dec: str, reason: str):
    """Overload para contact sheets del CSV completo."""
    old_dec = row.get("original_final_decision", row.get("decision", "?")) or "?"
    u3_raw  = row.get("quality_u3_decision_raw", row.get("u3_recalc_raw", "?")) or "?"
    p_good  = float(row.get("quality_u3_p_good", row.get("u3_recalc_p_good", 0.0)) or 0.0)
    p_bad   = float(row.get("quality_u3_p_bad",  row.get("u3_recalc_p_bad",  0.0)) or 0.0)
    return add_overlay(thumb, row.get("image", "?"), old_dec, new_dec,
                       u3_raw, p_good, p_bad, reason[:34])


def build_contact_sheet(entries, out_path: Path, title: str, cols=COLS):
    if not entries:
        blank = np.full((THUMB_H + 50, THUMB_W * cols, 3), 50, dtype=np.uint8)
        cv2.putText(blank, f"{title} - Sin imagenes", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        from PIL import Image
        Image.fromarray(cv2.cvtColor(blank, cv2.COLOR_BGR2RGB)).save(str(out_path), quality=90)
        print(f"  Contact sheet (vacio): {out_path}")
        return
    thumbs = []
    for t in entries:
        thumbs.append(t)
    rows_n = (len(thumbs) + cols - 1) // cols
    rows_i = []
    for r in range(rows_n):
        row_t = thumbs[r * cols: (r + 1) * cols]
        while len(row_t) < cols:
            row_t.append(np.full((THUMB_H, THUMB_W, 3), 30, dtype=np.uint8))
        rows_i.append(np.hstack(row_t))
    header = np.full((50, THUMB_W * cols, 3), 25, dtype=np.uint8)
    cv2.putText(header, title, (10, 33), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (220, 220, 220), 1, cv2.LINE_AA)
    sheet = np.vstack([header] + rows_i)
    from PIL import Image
    Image.fromarray(cv2.cvtColor(sheet, cv2.COLOR_BGR2RGB)).save(str(out_path), quality=90)
    print(f"  Contact sheet: {out_path}  ({len(entries)} peras)")


# ---------------------------------------------------------------------------
# TAREA 3: regla de decision para casos recalculados
# ---------------------------------------------------------------------------
def apply_tarea3_rule(u3_raw: str, p_good: float, p_bad: float) -> tuple:
    """
    Regla para casos donde U3 NO se habia usado antes.
    Sin YOLO de defectos activo -> strong_defect = False siempre.
    """
    if u3_raw == "U3_GOOD":
        if p_good >= 0.85:
            return "PASA", f"U3_GOOD_STRONG (p_good={p_good:.3f})"
        else:
            return "REVISAR", f"U3_GOOD_WEAK (p_good={p_good:.3f})"
    elif u3_raw == "U3_REVIEW":
        return "REVISAR", f"U3_REVIEW (p_bad={p_bad:.3f})"
    elif u3_raw == "U3_BAD":
        # Sin strong_defect_evidence -> REVISAR conservador
        return "REVISAR", f"U3_BAD_NO_STRONG_DEFECT_SAFE (p_bad={p_bad:.3f})"
    else:
        return "REVISAR", "U3_FAIL_SAFE_REVISAR"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import analyze_quality as aq

    # --- Cargar U3 ---
    if not U3_MODEL_PATH.exists():
        print(f"ERROR: modelo U3 no encontrado: {U3_MODEL_PATH}")
        sys.exit(1)
    u3_model, u3_thresholds = aq._load_u3_classifier(U3_MODEL_PATH, U3_THR_PATH)
    if u3_model is None:
        print("ERROR: no se pudo cargar U3")
        sys.exit(1)
    print(f"U3 cargado. bad_thr={u3_thresholds.get('bad_reject_threshold')}")

    # --- TAREA 2: Recalcular U3 para las 6 imagenes ---
    print(f"\n--- TAREA 2: Recalculo U3 para {len(NOT_USED_IMAGES)} imagenes ---")
    predictions = []
    for img_name in NOT_USED_IMAGES:
        img_path = find_image(img_name)
        row_status = {
            "image_name": img_name,
            "original_path": str(img_path) if img_path else "NOT_FOUND",
            "roi_status": "not_needed",
            "mask_status": "",
            "u3_status": "",
            "u3_pred": "",
            "u3_p_good": 0.0,
            "u3_p_bad": 0.0,
            "previous_decision": "REVISAR",
            "recommended_decision": "REVISAR",
            "reason": "",
        }

        if img_path is None:
            row_status["u3_status"] = "IMAGE_NOT_FOUND"
            row_status["reason"] = "Imagen no encontrada"
            predictions.append(row_status)
            print(f"  {img_name}: IMAGEN NO ENCONTRADA")
            continue

        # cv2.imread no soporta rutas Unicode en Windows; usar fromfile+imdecode
        try:
            raw = np.fromfile(str(img_path), dtype=np.uint8)
            bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        except Exception as e:
            bgr = None
        if bgr is None:
            row_status["u3_status"] = "READ_ERROR"
            row_status["reason"] = "no se pudo leer la imagen"
            predictions.append(row_status)
            print(f"  {img_name}: ERROR LECTURA")
            continue

        # Redimensionar si es muy grande (mismo limit que el pipeline: 1280)
        max_size = 1280
        h_i, w_i = bgr.shape[:2]
        if max(h_i, w_i) > max_size:
            scale = max_size / max(h_i, w_i)
            bgr = cv2.resize(bgr, (max(1, int(w_i * scale)), max(1, int(h_i * scale))),
                             interpolation=cv2.INTER_AREA)

        gray_pil = aq._make_u3_gray_input(bgr)
        if gray_pil is None:
            row_status["u3_status"] = "MASK_FAIL"
            row_status["mask_status"] = "gray_bg_fail"
            row_status["reason"] = "U3 no pudo generar mascara interna"
            row_status["recommended_decision"] = "REVISAR"
            predictions.append(row_status)
            print(f"  {img_name}: MASK_FAIL -> REVISAR")
            continue

        row_status["mask_status"] = "gray_bg_ok"
        p_bad, p_good, u3_raw = aq._run_u3_inference(u3_model, gray_pil, u3_thresholds)
        new_dec, reason = apply_tarea3_rule(u3_raw, p_good, p_bad)

        row_status.update({
            "u3_status": "OK",
            "u3_pred": "good" if p_good > p_bad else "bad",
            "u3_p_good": round(p_good, 4),
            "u3_p_bad": round(p_bad, 4),
            "recommended_decision": new_dec,
            "reason": reason,
        })
        predictions.append(row_status)
        print(f"  {img_name}: {u3_raw}  p_good={p_good:.3f}  p_bad={p_bad:.3f}  -> {new_dec}")

    # --- Guardar CSV de predicciones (TAREA 2) ---
    not_used_csv = NOT_USED_EVAL_DIR / "not_used_cases_predictions.csv"
    nu_fields = ["image_name","original_path","roi_status","mask_status",
                 "u3_status","u3_pred","u3_p_good","u3_p_bad",
                 "previous_decision","recommended_decision","reason"]
    with open(not_used_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=nu_fields)
        w.writeheader()
        for row in predictions:
            w.writerow({k: row.get(k, "") for k in nu_fields})
    print(f"\nCSV predicciones: {not_used_csv}")

    # --- Contact sheet de los 6 casos (TAREA 2) ---
    nu_thumbs = []
    for pred in predictions:
        ip = find_image(pred["image_name"])
        t = load_thumb(ip)
        t = add_overlay(t, pred["image_name"], "REVISAR",
                        pred["recommended_decision"],
                        pred.get("u3_status","?") if pred.get("u3_status") not in ("OK","") else
                        ("U3_GOOD" if pred["u3_p_good"] >= 0.55 else "U3_BAD"),
                        float(pred["u3_p_good"]), float(pred["u3_p_bad"]),
                        pred["reason"][:34])
        nu_thumbs.append(t)
    build_contact_sheet(nu_thumbs,
        NOT_USED_EVAL_DIR / "contact_sheet_not_used_cases.jpg",
        f"U3 Recalculado - {len(predictions)} casos no usados")

    # --- Summary TAREA 2 ---
    nu_summary_lines = ["U3 NOT USED CASES RECALCULO", "="*40, ""]
    for p in predictions:
        nu_summary_lines.append(
            f"  {p['image_name']}: U3={p.get('u3_status','')} "
            f"p_good={p['u3_p_good']:.3f} p_bad={p['u3_p_bad']:.3f} "
            f"-> {p['recommended_decision']}  ({p['reason']})")
    (NOT_USED_EVAL_DIR / "summary.txt").write_text("\n".join(nu_summary_lines), encoding="utf-8")

    # --- TAREA 4: Generar evaluacion v2 ---
    print(f"\n--- TAREA 4: Generando evaluacion fusion fixed v2 ---")
    if not INPUT_CSV.exists():
        print(f"ERROR: no se encuentra {INPUT_CSV}")
        sys.exit(1)

    # Construir mapa de predicciones recalculadas
    recalc_map = {p["image_name"]: p for p in predictions}

    all_rows_v2 = []
    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        original_fields = reader.fieldnames
        for row in reader:
            all_rows_v2.append(dict(row))

    # Actualizar filas de los 6 casos
    extra_fields = ["u3_recalc_status","u3_recalc_raw","u3_recalc_p_good","u3_recalc_p_bad",
                    "v2_decision","v2_reason"]
    v2_fields = list(original_fields) + [f for f in extra_fields if f not in original_fields]

    counts_before_v2 = {"PASA": 0, "REVISAR": 0, "RECHAZA": 0}
    counts_after_v2  = {"PASA": 0, "REVISAR": 0, "RECHAZA": 0}

    for row in all_rows_v2:
        old_dec = row.get("recommended_fixed_decision") or row.get("decision") or "REVISAR"
        counts_before_v2[old_dec] = counts_before_v2.get(old_dec, 0) + 1

        img_name = row.get("image", "")
        if img_name in recalc_map:
            pred = recalc_map[img_name]
            row["u3_recalc_status"]  = pred.get("u3_status", "")
            raw_guess = ("U3_GOOD" if pred["u3_p_good"] >= 0.55 else
                         ("U3_BAD" if pred["u3_p_bad"] >= 0.6 else "U3_REVIEW"))
            row["u3_recalc_raw"]     = raw_guess
            row["u3_recalc_p_good"]  = pred["u3_p_good"]
            row["u3_recalc_p_bad"]   = pred["u3_p_bad"]
            row["v2_decision"]       = pred["recommended_decision"]
            row["v2_reason"]         = pred["reason"]
            # Actualizar tambien quality_u3_* para que contact sheet los vea
            row["quality_u3_p_good"]       = pred["u3_p_good"]
            row["quality_u3_p_bad"]        = pred["u3_p_bad"]
            row["quality_u3_decision_raw"] = raw_guess
        else:
            row["u3_recalc_status"] = "not_recalculated"
            row["u3_recalc_raw"]    = row.get("quality_u3_decision_raw", "")
            row["u3_recalc_p_good"] = row.get("quality_u3_p_good", 0.0)
            row["u3_recalc_p_bad"]  = row.get("quality_u3_p_bad", 0.0)
            row["v2_decision"] = row.get("recommended_fixed_decision", row.get("decision", "REVISAR"))
            row["v2_reason"]   = row.get("recommended_reason", "")

        counts_after_v2[row["v2_decision"]] = counts_after_v2.get(row["v2_decision"], 0) + 1

    # Guardar CSV v2
    csv_v2 = V2_DIR / "resultados_u3_fusion_fixed_v2.csv"
    with open(csv_v2, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=v2_fields, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows_v2:
            writer.writerow(row)
    print(f"CSV v2: {csv_v2}")

    # --- Contact sheets v2 ---
    entry_all_v2    = []
    entry_revbad_v2 = []

    for row in all_rows_v2:
        img_path = find_image(row.get("image", ""))
        new_dec  = row["v2_decision"]
        reason   = row["v2_reason"]
        t = load_thumb(img_path)
        t = add_overlay_from_row(t, row, new_dec, reason)
        entry_all_v2.append(t)
        if new_dec in ("REVISAR", "RECHAZA"):
            entry_revbad_v2.append(t)

    title_all_v2 = (
        f"U3 Fusion Fixed v2 - Todas ({len(entry_all_v2)})  "
        f"PASA={counts_after_v2['PASA']}  "
        f"REVISAR={counts_after_v2['REVISAR']}  "
        f"RECHAZA={counts_after_v2['RECHAZA']}"
    )
    build_contact_sheet(entry_all_v2,
        V2_DIR / "contact_sheet_u3_fusion_fixed_v2_all.jpg", title_all_v2)
    build_contact_sheet(entry_revbad_v2,
        V2_DIR / "contact_sheet_u3_fusion_fixed_v2_review_bad.jpg",
        f"U3 Fusion Fixed v2 - REVISAR+RECHAZA ({len(entry_revbad_v2)})")

    # --- Summary v2 ---
    rechaza_recalc = 0
    revisar_recalc = 0
    pasa_recalc = 0
    cases_detail = []
    for pred in predictions:
        d = pred["recommended_decision"]
        if d == "PASA": pasa_recalc += 1
        elif d == "REVISAR": revisar_recalc += 1
        elif d == "RECHAZA": rechaza_recalc += 1
        cases_detail.append(
            f"  {pred['image_name']}: {pred.get('u3_status','')} "
            f"p_good={pred['u3_p_good']:.3f} -> {d}  ({pred['reason']})")

    summary_lines = [
        "U3 FUSION FIXED v2 - RESUMEN",
        "=" * 50,
        "",
        "ANTES (fusion fixed v1):",
        f"  Total : {len(all_rows_v2)}",
        f"  PASA  : {counts_before_v2.get('PASA', 0)}",
        f"  REVISAR: {counts_before_v2.get('REVISAR', 0)}",
        f"  RECHAZA: {counts_before_v2.get('RECHAZA', 0)}",
        "",
        "DESPUES (fusion fixed v2 con recalculo U3):",
        f"  Total : {len(all_rows_v2)}",
        f"  PASA  : {counts_after_v2.get('PASA', 0)}",
        f"  REVISAR: {counts_after_v2.get('REVISAR', 0)}",
        f"  RECHAZA: {counts_after_v2.get('RECHAZA', 0)}",
        "",
        "Casos recalculados (6 imagenes U3_NOT_USED):",
    ] + cases_detail + [
        "",
        "Archivos:",
        f"  {NOT_USED_EVAL_DIR / 'not_used_cases_predictions.csv'}",
        f"  {NOT_USED_EVAL_DIR / 'contact_sheet_not_used_cases.jpg'}",
        f"  {csv_v2}",
        f"  {V2_DIR / 'contact_sheet_u3_fusion_fixed_v2_all.jpg'}",
        f"  {V2_DIR / 'contact_sheet_u3_fusion_fixed_v2_review_bad.jpg'}",
    ]
    (V2_DIR / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    print()
    for line in summary_lines:
        print(line)

    # --- Reporte final (TAREA 5) ---
    all_pasa_count  = counts_after_v2.get("PASA", 0)
    all_rev_count   = counts_after_v2.get("REVISAR", 0)
    all_rec_count   = counts_after_v2.get("RECHAZA", 0)
    detail_md = "\n".join(
        f"| {p['image_name']} | {p['u3_status']} | {p['u3_p_good']:.3f} | {p['u3_p_bad']:.3f} | {p['recommended_decision']} | {p['reason']} |"
        for p in predictions)

    fix_report = f"""# Reporte: Corrección casos U3_NOT_USED — u3_not_used_cases_fix_report

**Fecha:** 2026-05-21
**Base:** `outputs/u3_fusion_fixed_eval/resultados_u3_fusion_fixed.csv` (86 filas, v1)

---

## Antes

| Decisión | N |
|---|---|
| PASA | {counts_before_v2.get('PASA', 0)} |
| REVISAR | {counts_before_v2.get('REVISAR', 0)} |
| RECHAZA | {counts_before_v2.get('RECHAZA', 0)} |
| **Total** | **{len(all_rows_v2)}** |

## Después

| Decisión | N |
|---|---|
| PASA | {all_pasa_count} |
| REVISAR | {all_rev_count} |
| RECHAZA | {all_rec_count} |
| **Total** | **{len(all_rows_v2)}** |

---

## Resultados por imagen recalculada

| Imagen | U3 status | p_good | p_bad | Decisión | Razón |
|---|---|---|---|---|---|
{detail_md}

---

## Validación (TAREA 5)

- Total = {len(all_rows_v2)} ({'OK' if len(all_rows_v2) == 86 else 'ERROR'})
- RECHAZA = {all_rec_count} ({'OK' if all_rec_count == 0 else 'REVISAR'})
- Las 6 imágenes recalculadas tienen p_good real (ya no g=0.00 b=0.00): {'OK' if all(float(p['u3_p_good']) > 0 for p in predictions if p['u3_status'] == 'OK') else 'ALGUNOS FALLARON'}

---

## Qué NO se modificó

- Modelo U3 sin cambios
- quality_rules.yaml sin cambios
- V2 sin cambios
- Outputs anteriores sin borrar

---

## Archivos generados

| Archivo | Estado |
|---|---|
| `outputs/u3_not_used_cases_eval/not_used_cases_predictions.csv` | generado |
| `outputs/u3_not_used_cases_eval/contact_sheet_not_used_cases.jpg` | generado |
| `outputs/u3_not_used_cases_eval/summary.txt` | generado |
| `outputs/u3_fusion_fixed_eval_v2/resultados_u3_fusion_fixed_v2.csv` | generado |
| `outputs/u3_fusion_fixed_eval_v2/contact_sheet_u3_fusion_fixed_v2_all.jpg` | generado |
| `outputs/u3_fusion_fixed_eval_v2/contact_sheet_u3_fusion_fixed_v2_review_bad.jpg` | generado |
| `outputs/u3_fusion_fixed_eval_v2/summary.txt` | generado |
"""
    (REPORTS_DIR / "u3_not_used_cases_fix_report.md").write_text(fix_report, encoding="utf-8")
    print(f"\nReporte: {REPORTS_DIR / 'u3_not_used_cases_fix_report.md'}")

    return counts_after_v2, predictions


if __name__ == "__main__":
    main()
