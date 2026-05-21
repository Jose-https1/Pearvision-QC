"""
reevaluate_u3_fusion_fixed.py

Aplica la logica de fusion U3 corregida al CSV existente sin reejecutar el
pipeline completo ni reentrenar ningun modelo.

Input:  outputs/u3_integrated_pipeline_eval/resultados_integrated_u3.csv
Output: outputs/u3_fusion_fixed_eval/
"""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_DIRS = [
    PROJECT_ROOT / "data/unseen_quality_eval_input/supermarket_valid_conditions_batch_v3",
    PROJECT_ROOT / "data/unseen_quality_eval_input/supermarket_good_batch_v2",
    PROJECT_ROOT / "data/supermarket_good_hard_examples_v1/images",
    PROJECT_ROOT / "data/supermarket_good_hard_examples_v2/images",
]

INPUT_CSV = PROJECT_ROOT / "outputs/u3_integrated_pipeline_eval/resultados_integrated_u3.csv"
OUT_DIR = PROJECT_ROOT / "outputs/u3_fusion_fixed_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DECISION_COLOR = {"PASA": (0, 200, 0), "REVISAR": (0, 165, 255), "RECHAZA": (0, 0, 220)}
THUMB_W, THUMB_H = 280, 280
COLS = 5


# ---------------------------------------------------------------------------
# TAREA 3: strong_defect_evidence
# ---------------------------------------------------------------------------
def compute_strong_defect_evidence(row: dict) -> bool:
    """
    True solo si hay evidencia fuerte de defecto visible real.
    NO considera: russeting, color marron uniforme, piel rugosa, sombra,
    fondo, cambio de iluminacion, caliz/pedunculo.
    NO considera la decision rule-based RECHAZA como evidencia suficiente
    (esa es exactamente la categoria de falso rechazo que estamos corrigiendo).
    """
    yolo_count = int(float(row.get("yolo_defect_count", 0) or 0))
    yolo_conf = float(row.get("yolo_defect_max_conf", 0.0) or 0.0)
    dark_rot_pct = float(row.get("dark_rot_pct", 0.0) or 0.0)
    body_l_mean = float(row.get("body_l_mean", 128.0) or 128.0)

    has_yolo_defect = (yolo_count >= 2 or yolo_conf > 0.65)
    has_real_necrosis = (dark_rot_pct > 50.0 and body_l_mean < 45.0)

    return has_yolo_defect or has_real_necrosis


# ---------------------------------------------------------------------------
# TAREA 4: nueva logica de fusion
# ---------------------------------------------------------------------------
def apply_new_fusion(row: dict) -> tuple:
    """
    Aplica la fusion corregida. Retorna (new_decision, reason).

    Logica:
    - U3 no usado (captura invalida, error) -> mantener original
    - U3=GOOD + p_good>=0.85 + sin defecto fuerte -> PASA
    - U3=GOOD + p_good>=0.85 + defecto fuerte     -> REVISAR
    - U3=GOOD + 0.55<=p_good<0.85 + era RECHAZA + sin defecto -> REVISAR
    - U3=GOOD + 0.55<=p_good<0.85 + era PASA      -> PASA
    - U3=GOOD + 0.55<=p_good<0.85 + resto          -> REVISAR
    - U3=REVIEW                                     -> REVISAR
    - U3=BAD  + defecto fuerte                      -> RECHAZA
    - U3=BAD  + sin defecto fuerte                  -> REVISAR
    """
    u3_status = row.get("quality_u3_status", "not_used")
    u3_raw = row.get("quality_u3_decision_raw", "")
    p_good = float(row.get("quality_u3_p_good", 0.0) or 0.0)
    p_bad = float(row.get("quality_u3_p_bad", 0.0) or 0.0)
    original_decision = row.get("decision", "REVISAR")
    decision_before_u3 = row.get("final_decision_before_u3", original_decision) or original_decision

    if u3_status in ("not_used", "") or not u3_raw:
        return original_decision, "U3_NOT_USED_KEEP_ORIGINAL"
    if u3_status == "ERROR" or u3_raw == "U3_ERROR":
        return original_decision, "U3_ERROR_KEEP_ORIGINAL"
    if u3_status == "MASK_FAIL":
        return "REVISAR" if original_decision == "PASA" else original_decision, "U3_MASK_FAIL"

    strong = compute_strong_defect_evidence(row)

    if u3_raw == "U3_GOOD":
        if p_good >= 0.85:
            if not strong:
                return "PASA", f"U3_GOOD_STRONG_NO_STRONG_DEFECT (p_good={p_good:.3f})"
            else:
                return "REVISAR", f"U3_GOOD_BUT_STRONG_DEFECT_REVIEW (p_good={p_good:.3f})"
        else:  # 0.55 <= p_good < 0.85
            if decision_before_u3 == "RECHAZA" and not strong:
                return "REVISAR", f"U3_GOOD_WEAK_PROTECT_FROM_REJECT (p_good={p_good:.3f})"
            elif decision_before_u3 == "PASA":
                return "PASA", f"U3_GOOD_WEAK_WAS_PASA (p_good={p_good:.3f})"
            else:
                return "REVISAR", f"U3_GOOD_WEAK_REVISAR (p_good={p_good:.3f})"

    elif u3_raw == "U3_REVIEW":
        return "REVISAR", f"U3_REVIEW (p_bad={p_bad:.3f})"

    elif u3_raw == "U3_BAD":
        if strong:
            return "RECHAZA", f"U3_BAD_AND_STRONG_DEFECT (p_bad={p_bad:.3f})"
        else:
            return "REVISAR", f"U3_BAD_WITHOUT_STRONG_DEFECT_SAFE_REVIEW (p_bad={p_bad:.3f})"

    return original_decision, "U3_UNKNOWN_KEEP_ORIGINAL"


# ---------------------------------------------------------------------------
# Utilidades de imagen
# ---------------------------------------------------------------------------
def find_image(image_name: str) -> Path:
    for src in SOURCE_DIRS:
        cand = src / image_name
        if cand.exists():
            return cand
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


def add_overlay(thumb, row: dict, new_decision: str, reason: str, strong: bool):
    """Borde de color + etiqueta multi-linea con toda la info requerida."""
    color = DECISION_COLOR.get(new_decision, (200, 200, 200))
    cv2.rectangle(thumb, (0, 0), (thumb.shape[1]-1, thumb.shape[0]-1), color, 5)

    old_decision = row.get("decision", "?")
    p_good = float(row.get("quality_u3_p_good", 0.0) or 0.0)
    p_bad = float(row.get("quality_u3_p_bad", 0.0) or 0.0)
    u3_raw = row.get("quality_u3_decision_raw", "?") or "?"
    img_name = (row.get("image", "?") or "?")[:20]

    lines = [
        img_name,
        f"ANT:{old_decision[:7]}  NOW:{new_decision}",
        f"U3:{u3_raw}  g={p_good:.2f} b={p_bad:.2f}",
        f"SDE:{str(strong)[0]}  {reason[:26]}",
    ]

    overlay_h = len(lines) * 18 + 8
    h, w = thumb.shape[:2]
    cv2.rectangle(thumb, (0, h - overlay_h), (w, h), (15, 15, 15), -1)
    for i, line in enumerate(lines):
        y = h - overlay_h + 15 + i * 17
        cv2.putText(thumb, line, (3, y), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(thumb, line, (3, y), cv2.FONT_HERSHEY_SIMPLEX, 0.37, color, 1, cv2.LINE_AA)
    return thumb


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
    for img_path, row, new_decision, reason, strong in entries:
        t = load_thumb(img_path)
        t = add_overlay(t, row, new_decision, reason, strong)
        thumbs.append(t)

    rows_count = (len(thumbs) + cols - 1) // cols
    rows_imgs = []
    for r_idx in range(rows_count):
        row_thumbs = thumbs[r_idx * cols: (r_idx + 1) * cols]
        while len(row_thumbs) < cols:
            row_thumbs.append(np.full((THUMB_H, THUMB_W, 3), 30, dtype=np.uint8))
        rows_imgs.append(np.hstack(row_thumbs))

    header = np.full((50, THUMB_W * cols, 3), 25, dtype=np.uint8)
    cv2.putText(header, title, (10, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1, cv2.LINE_AA)

    sheet = np.vstack([header] + rows_imgs)
    from PIL import Image
    Image.fromarray(cv2.cvtColor(sheet, cv2.COLOR_BGR2RGB)).save(str(out_path), quality=90)
    print(f"  Contact sheet: {out_path}  ({len(entries)} peras)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if not INPUT_CSV.exists():
        print(f"ERROR: no se encuentra {INPUT_CSV}")
        sys.exit(1)

    rows = []
    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))

    print(f"CSV cargado: {len(rows)} filas desde {INPUT_CSV.name}")

    # --- Aplicar nueva fusion ---
    results = []
    for row in rows:
        new_decision, reason = apply_new_fusion(row)
        strong = compute_strong_defect_evidence(row)
        img_path = find_image(row.get("image", ""))
        results.append({
            "row": row,
            "new_decision": new_decision,
            "reason": reason,
            "strong_defect_evidence": strong,
            "img_path": img_path,
        })

    # --- Conteos ---
    counts_before = {"PASA": 0, "REVISAR": 0, "RECHAZA": 0}
    counts_after  = {"PASA": 0, "REVISAR": 0, "RECHAZA": 0}
    rechaza_to_pasa = 0
    rechaza_to_revisar = 0
    rechaza_stayed = 0
    revisar_to_pasa = 0

    for r in results:
        old = r["row"].get("decision", "REVISAR") or "REVISAR"
        new = r["new_decision"]
        counts_before[old] = counts_before.get(old, 0) + 1
        counts_after[new]  = counts_after.get(new, 0) + 1
        if old == "RECHAZA":
            if new == "PASA":
                rechaza_to_pasa += 1
            elif new == "REVISAR":
                rechaza_to_revisar += 1
            else:
                rechaza_stayed += 1
        if old == "REVISAR" and new == "PASA":
            revisar_to_pasa += 1

    # --- CSV enriquecido (TAREA 2) ---
    csv_out = OUT_DIR / "resultados_u3_fusion_fixed.csv"
    base_keys = list(rows[0].keys())
    new_keys = [
        "strong_defect_evidence",
        "original_final_decision",
        "final_decision_before_u3_inferred",
        "recommended_fixed_decision",
        "recommended_reason",
        "fusion_problem_detected",
        "rejection_source",
    ]
    fieldnames_out = base_keys + [k for k in new_keys if k not in base_keys]

    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_out, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row_out = dict(r["row"])
            old = row_out.get("decision", "")
            row_out["strong_defect_evidence"] = r["strong_defect_evidence"]
            row_out["original_final_decision"] = old
            row_out["final_decision_before_u3_inferred"] = row_out.get("final_decision_before_u3", old)
            row_out["recommended_fixed_decision"] = r["new_decision"]
            row_out["recommended_reason"] = r["reason"]
            row_out["fusion_problem_detected"] = (old != r["new_decision"])
            row_out["rejection_source"] = (
                "rule_based_color_texture" if old == "RECHAZA" else
                "capture_invalid" if row_out.get("capture_valid") == "False" else ""
            )
            writer.writerow(row_out)
    print(f"\nCSV enriquecido guardado: {csv_out}")

    # --- Contact sheets (TAREA 7) ---
    entry_all    = []
    entry_revbad = []
    for r in results:
        entry = (r["img_path"], r["row"], r["new_decision"], r["reason"], r["strong_defect_evidence"])
        entry_all.append(entry)
        if r["new_decision"] in ("REVISAR", "RECHAZA"):
            entry_revbad.append(entry)

    title_all = (
        f"U3 Fusion Fixed - Todas ({len(entry_all)})  "
        f"PASA={counts_after['PASA']}  REVISAR={counts_after['REVISAR']}  RECHAZA={counts_after['RECHAZA']}"
    )
    sheet_all     = OUT_DIR / "contact_sheet_u3_fusion_fixed_all.jpg"
    sheet_revbad  = OUT_DIR / "contact_sheet_u3_fusion_fixed_review_bad.jpg"

    print(f"\nGenerando contact sheets...")
    build_contact_sheet(entry_all, sheet_all, title_all)
    build_contact_sheet(entry_revbad, sheet_revbad,
        f"U3 Fusion Fixed - REVISAR+RECHAZA ({len(entry_revbad)})")

    # --- Summary (TAREA 6) ---
    summary_lines = [
        "U3 FUSION FIX - RESUMEN",
        "=" * 50,
        "",
        "ANTES (pipeline integrado con bug de fusion):",
        f"  Total evaluadas : {len(rows)}",
        f"  PASA            : {counts_before.get('PASA', 0)}",
        f"  REVISAR         : {counts_before.get('REVISAR', 0)}",
        f"  RECHAZA         : {counts_before.get('RECHAZA', 0)}",
        "",
        "DESPUES (fusion corregida):",
        f"  Total evaluadas : {len(rows)}",
        f"  PASA            : {counts_after.get('PASA', 0)}",
        f"  REVISAR         : {counts_after.get('REVISAR', 0)}",
        f"  RECHAZA         : {counts_after.get('RECHAZA', 0)}",
        "",
        "Correcciones:",
        f"  RECHAZA -> PASA    : {rechaza_to_pasa}",
        f"  RECHAZA -> REVISAR : {rechaza_to_revisar}",
        f"  RECHAZA mantenidas : {rechaza_stayed}",
        f"  REVISAR -> PASA    : {revisar_to_pasa}",
        "",
        "Objetivo (peras sanas de supermercado):",
        f"  RECHAZA = {counts_after.get('RECHAZA', 0)}  (objetivo: 0)",
        f"  PASA    = {counts_after.get('PASA', 0)}/86  (objetivo: >=60)",
        "",
        "Archivos generados:",
        f"  {csv_out}",
        f"  {sheet_all}",
        f"  {sheet_revbad}",
        f"  {OUT_DIR / 'summary.txt'}",
    ]
    summary_path = OUT_DIR / "summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print()
    for line in summary_lines:
        print(line)

    return {
        "total": len(rows),
        "before": counts_before,
        "after": counts_after,
        "rechaza_to_pasa": rechaza_to_pasa,
        "rechaza_to_revisar": rechaza_to_revisar,
        "rechaza_stayed": rechaza_stayed,
        "revisar_to_pasa": revisar_to_pasa,
    }


if __name__ == "__main__":
    main()
