"""
evaluate_final_u3_pipeline_bad_regression_v1.py

Evalua el pipeline U3 fusion (fusión corregida v2) sobre el dataset humano
quality_fruits360_human_v1 (49 GOOD + 220 BAD).

Objetivo: verificar que la corrección que logra 86/86 PASA en peras sanas de
supermercado no ha creado falsos aceptados (BAD -> PASA) en peras con defectos.

No entrena ningun modelo. No modifica U3, V2 ni quality_rules.yaml.
"""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

DATASET_CSV  = PROJECT_ROOT / "data/quality_fruits360_human_v1/metadata/quality_fruits360_human_v1_master.csv"
DATASET_ROOT = PROJECT_ROOT / "data/quality_fruits360_human_v1"
U3_MODEL     = PROJECT_ROOT / "outputs/fruits360_quality_cls_u3_roi_masked_clean/best_model.pt"
U3_THR       = PROJECT_ROOT / "outputs/fruits360_quality_cls_u3_roi_masked_clean/selected_thresholds.json"

OUT_DIR      = PROJECT_ROOT / "outputs/final_u3_bad_regression_eval"
REPORTS_DIR  = PROJECT_ROOT / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Contact-sheet layout
THUMB_W, THUMB_H = 150, 150
COLS = 8

# Colores BGR para contact sheets
COLOR_OK       = (0, 200, 0)     # verde  = correcto
COLOR_REVIEW   = (0, 165, 255)   # naranja = revisar
COLOR_CRITICAL = (0, 0, 220)     # rojo   = error critico


# ---------------------------------------------------------------------------
# Logica de fusion (igual que v2, sin YOLO detector activo)
# ---------------------------------------------------------------------------
def compute_strong_defect_evidence(yolo_count: int = 0, yolo_conf: float = 0.0,
                                   dark_rot_pct: float = 0.0, body_l_mean: float = 128.0) -> bool:
    has_yolo   = (yolo_count >= 2 or yolo_conf > 0.65)
    has_real_rot = (dark_rot_pct > 50.0 and body_l_mean < 45.0)
    return has_yolo or has_real_rot


def apply_fusion(u3_raw: str, p_good: float, p_bad: float,
                 strong: bool) -> tuple:
    """Retorna (decision, reason)."""
    if not u3_raw or u3_raw == "U3_ERROR":
        return "REVISAR", "U3_ERROR_SAFE"

    if u3_raw == "U3_GOOD":
        if p_good >= 0.85:
            if not strong:
                return "PASA",    f"U3_GOOD_STRONG_NO_DEFECT (p={p_good:.3f})"
            else:
                return "REVISAR", f"U3_GOOD_STRONG_DEFECT_REVIEW (p={p_good:.3f})"
        else:  # 0.55 <= p_good < 0.85
            return "REVISAR", f"U3_GOOD_WEAK (p={p_good:.3f})"

    elif u3_raw == "U3_REVIEW":
        return "REVISAR", f"U3_REVIEW (p_bad={p_bad:.3f})"

    elif u3_raw == "U3_BAD":
        if strong:
            return "RECHAZA", f"U3_BAD_STRONG_DEFECT (p_bad={p_bad:.3f})"
        else:
            return "REVISAR",  f"U3_BAD_NO_STRONG_DEFECT_SAFE (p_bad={p_bad:.3f})"

    return "REVISAR", "UNKNOWN_SAFE"


def business_result(human_label: str, decision: str) -> str:
    if human_label == "GOOD":
        if decision == "PASA":    return "GOOD_PASS_OK"
        if decision == "REVISAR": return "GOOD_REVIEW_OK_CONSERVATIVE"
        if decision == "RECHAZA": return "GOOD_REJECT_CRITICAL_FALSE_REJECT"
    if human_label == "BAD":
        if decision == "RECHAZA": return "BAD_REJECT_OK"
        if decision == "REVISAR": return "BAD_REVIEW_OK_CONSERVATIVE"
        if decision == "PASA":    return "BAD_PASS_CRITICAL_FALSE_ACCEPT"
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Helpers de imagen
# ---------------------------------------------------------------------------
def imread_unicode(path: Path) -> np.ndarray:
    """cv2.imread seguro para rutas Unicode en Windows."""
    try:
        raw = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(raw, cv2.IMREAD_COLOR)
    except Exception:
        return None


def load_thumb(img_path, w=THUMB_W, h=THUMB_H) -> np.ndarray:
    if img_path is None or not Path(str(img_path)).exists():
        return np.full((h, w, 3), 70, dtype=np.uint8)
    try:
        from PIL import Image as PILImage
        pil = PILImage.open(str(img_path)).convert("RGB")
        pil = pil.resize((w, h))
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        return np.full((h, w, 3), 70, dtype=np.uint8)


def add_overlay(thumb: np.ndarray, row: dict, border_color) -> np.ndarray:
    cv2.rectangle(thumb, (0, 0), (thumb.shape[1]-1, thumb.shape[0]-1), border_color, 4)
    human  = row.get("human_label", "?")
    dec    = row.get("final_decision", "?")
    p_good = float(row.get("u3_p_good", 0.0))
    p_bad  = float(row.get("u3_p_bad", 0.0))
    fname  = (row.get("image_name", "?") or "?")[:18]
    reason_short = (row.get("reason", "") or "")[:22]

    lines = [fname, f"{human}->{dec}", f"g={p_good:.2f} b={p_bad:.2f}", reason_short]
    ov_h = len(lines) * 16 + 6
    h, w = thumb.shape[:2]
    cv2.rectangle(thumb, (0, h - ov_h), (w, h), (15, 15, 15), -1)
    for i, line in enumerate(lines):
        y = h - ov_h + 13 + i * 15
        cv2.putText(thumb, line, (2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(thumb, line, (2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.33, border_color, 1, cv2.LINE_AA)
    return thumb


def decide_color(biz_result: str):
    if "CRITICAL" in biz_result:
        return COLOR_CRITICAL
    if "CONSERVATIVE" in biz_result or biz_result == "UNKNOWN":
        return COLOR_REVIEW
    return COLOR_OK


def build_contact_sheet(entries: list, out_path: Path, title: str, cols=COLS):
    if not entries:
        blank = np.full((THUMB_H + 50, THUMB_W * min(cols, 4), 3), 40, dtype=np.uint8)
        cv2.putText(blank, f"{title} (sin casos)", (6, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        from PIL import Image as PILImage
        PILImage.fromarray(cv2.cvtColor(blank, cv2.COLOR_BGR2RGB)).save(str(out_path), quality=90)
        print(f"  Sheet (vacio): {out_path.name}")
        return
    thumbs = []
    for img_path, row in entries:
        biz = row.get("business_result", "UNKNOWN")
        color = decide_color(biz)
        t = load_thumb(img_path)
        t = add_overlay(t, row, color)
        thumbs.append(t)
    rows_n = (len(thumbs) + cols - 1) // cols
    rows_i = []
    for r in range(rows_n):
        row_t = thumbs[r * cols: (r+1) * cols]
        while len(row_t) < cols:
            row_t.append(np.full((THUMB_H, THUMB_W, 3), 30, dtype=np.uint8))
        rows_i.append(np.hstack(row_t))
    header = np.full((46, THUMB_W * cols, 3), 22, dtype=np.uint8)
    cv2.putText(header, title, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.60,
                (220, 220, 220), 1, cv2.LINE_AA)
    sheet = np.vstack([header] + rows_i)
    from PIL import Image as PILImage
    PILImage.fromarray(cv2.cvtColor(sheet, cv2.COLOR_BGR2RGB)).save(str(out_path), quality=90)
    print(f"  Sheet: {out_path.name}  ({len(entries)} imgs)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import analyze_quality as aq

    # Cargar U3
    if not U3_MODEL.exists():
        print(f"ERROR: U3 no encontrado: {U3_MODEL}")
        sys.exit(1)
    u3_model, u3_thr = aq._load_u3_classifier(U3_MODEL, U3_THR)
    if u3_model is None:
        print("ERROR: no se pudo cargar U3")
        sys.exit(1)
    print(f"U3 cargado  bad_thr={u3_thr['bad_reject_threshold']}  good_thr={u3_thr['good_accept_threshold']}")

    # Leer dataset CSV
    dataset_rows = []
    with open(DATASET_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            dataset_rows.append(row)
    print(f"Dataset: {len(dataset_rows)} filas")

    # Evaluar cada imagen
    results = []
    n_good = n_bad = 0
    for i, row in enumerate(dataset_rows):
        human_label = row["human_label"]
        fname   = row["filename"]
        split_  = row["split"]
        class_  = row["class"]
        img_path = DATASET_ROOT / split_ / class_ / fname

        if human_label == "GOOD": n_good += 1
        else:                     n_bad  += 1

        # Leer imagen (Unicode-safe)
        bgr = imread_unicode(img_path)
        if bgr is None:
            result_row = {
                "image_name": fname, "human_label": human_label,
                "image_path": str(img_path), "split": split_, "class": class_,
                "u3_status": "READ_ERROR", "u3_pred": "", "u3_p_good": 0.0, "u3_p_bad": 0.0,
                "u3_raw": "", "strong_defect_evidence": False,
                "final_decision": "REVISAR", "reason": "READ_ERROR",
                "business_result": business_result(human_label, "REVISAR"),
            }
            results.append(result_row)
            print(f"  [{i+1}/{len(dataset_rows)}] {fname}: READ_ERROR")
            continue

        # Generar gray_bg_clean para U3
        gray_pil = aq._make_u3_gray_input(bgr)
        if gray_pil is None:
            result_row = {
                "image_name": fname, "human_label": human_label,
                "image_path": str(img_path), "split": split_, "class": class_,
                "u3_status": "MASK_FAIL", "u3_pred": "", "u3_p_good": 0.0, "u3_p_bad": 0.0,
                "u3_raw": "", "strong_defect_evidence": False,
                "final_decision": "REVISAR", "reason": "U3_MASK_FAIL",
                "business_result": business_result(human_label, "REVISAR"),
            }
            results.append(result_row)
            continue

        # Inferencia U3
        p_bad, p_good, u3_raw = aq._run_u3_inference(u3_model, gray_pil, u3_thr)
        strong = compute_strong_defect_evidence()  # sin YOLO -> siempre False

        decision, reason = apply_fusion(u3_raw, p_good, p_bad, strong)
        biz = business_result(human_label, decision)

        result_row = {
            "image_name": fname,
            "human_label": human_label,
            "image_path": str(img_path),
            "split": split_,
            "class": class_,
            "u3_status": "OK",
            "u3_pred": "good" if p_good > p_bad else "bad",
            "u3_p_good": round(p_good, 4),
            "u3_p_bad": round(p_bad, 4),
            "u3_raw": u3_raw,
            "strong_defect_evidence": strong,
            "final_decision": decision,
            "reason": reason,
            "business_result": biz,
        }
        results.append(result_row)

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(dataset_rows)}] procesadas...")

    print(f"\nEvaluacion completa: {len(results)} imagenes")

    # --- Contar metricas ---
    counts = {
        "GOOD_PASS_OK": 0, "GOOD_REVIEW_OK_CONSERVATIVE": 0, "GOOD_REJECT_CRITICAL_FALSE_REJECT": 0,
        "BAD_REJECT_OK": 0, "BAD_REVIEW_OK_CONSERVATIVE": 0, "BAD_PASS_CRITICAL_FALSE_ACCEPT": 0,
    }
    for r in results:
        biz = r["business_result"]
        counts[biz] = counts.get(biz, 0) + 1

    total = len(results)
    good_total = counts["GOOD_PASS_OK"] + counts["GOOD_REVIEW_OK_CONSERVATIVE"] + counts["GOOD_REJECT_CRITICAL_FALSE_REJECT"]
    bad_total  = counts["BAD_REJECT_OK"] + counts["BAD_REVIEW_OK_CONSERVATIVE"] + counts["BAD_PASS_CRITICAL_FALSE_ACCEPT"]
    total_pasa    = counts["GOOD_PASS_OK"] + counts["BAD_PASS_CRITICAL_FALSE_ACCEPT"]
    total_revisar = counts["GOOD_REVIEW_OK_CONSERVATIVE"] + counts["BAD_REVIEW_OK_CONSERVATIVE"]
    total_rechaza = counts["GOOD_REJECT_CRITICAL_FALSE_REJECT"] + counts["BAD_REJECT_OK"]

    false_reject_rate = counts["GOOD_REJECT_CRITICAL_FALSE_REJECT"] / max(good_total, 1)
    false_accept_rate = counts["BAD_PASS_CRITICAL_FALSE_ACCEPT"]   / max(bad_total,  1)
    auto_accept_rate  = total_pasa    / total
    manual_review_rate= total_revisar / total
    reject_rate       = total_rechaza / total

    # --- Guardar CSV ---
    csv_out = OUT_DIR / "results_final_u3_bad_regression.csv"
    fields = ["image_name","human_label","image_path","split","class",
              "u3_status","u3_pred","u3_p_good","u3_p_bad","u3_raw",
              "strong_defect_evidence","final_decision","reason","business_result"]
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k,"") for k in fields})
    print(f"CSV: {csv_out}")

    # --- Contact sheets ---
    # Separar por tipo
    entry_all            = []
    entry_bad_pass       = []
    entry_good_reject    = []
    entry_review         = []

    for r in results:
        img_p = Path(r["image_path"])
        biz   = r["business_result"]
        entry_all.append((img_p, r))
        if biz == "BAD_PASS_CRITICAL_FALSE_ACCEPT":
            entry_bad_pass.append((img_p, r))
        elif biz == "GOOD_REJECT_CRITICAL_FALSE_REJECT":
            entry_good_reject.append((img_p, r))
        if "CONSERVATIVE" in biz:
            entry_review.append((img_p, r))

    print(f"\nGenerando contact sheets...")
    title_all = (
        f"Bad Regression Eval ({total}) | "
        f"PASA={total_pasa} REV={total_revisar} REC={total_rechaza} | "
        f"FRR={false_reject_rate:.1%} FAR={false_accept_rate:.1%}"
    )
    build_contact_sheet(entry_all, OUT_DIR / "contact_sheet_all.jpg", title_all)
    build_contact_sheet(entry_bad_pass,
        OUT_DIR / "contact_sheet_bad_pass_critical.jpg",
        f"ERRORES CRITICOS: BAD -> PASA ({len(entry_bad_pass)})")
    build_contact_sheet(entry_good_reject,
        OUT_DIR / "contact_sheet_good_reject_critical.jpg",
        f"ERRORES CRITICOS: GOOD -> RECHAZA ({len(entry_good_reject)})")
    build_contact_sheet(entry_review,
        OUT_DIR / "contact_sheet_review_cases.jpg",
        f"REVISABLES ({len(entry_review)}): BAD->REVISAR + GOOD->REVISAR")

    # --- Summary ---
    summary_lines = [
        "FINAL U3 BAD REGRESSION EVAL — RESUMEN",
        "=" * 55, "",
        f"Dataset:  quality_fruits360_human_v1  ({total} imgs)",
        f"  GOOD: {good_total}   BAD: {bad_total}", "",
        "Resultados:",
        f"  GOOD -> PASA    (OK)          : {counts['GOOD_PASS_OK']}",
        f"  GOOD -> REVISAR (conservador) : {counts['GOOD_REVIEW_OK_CONSERVATIVE']}",
        f"  GOOD -> RECHAZA (CRITICO)     : {counts['GOOD_REJECT_CRITICAL_FALSE_REJECT']}",
        f"  BAD  -> RECHAZA (OK)          : {counts['BAD_REJECT_OK']}",
        f"  BAD  -> REVISAR (conservador) : {counts['BAD_REVIEW_OK_CONSERVATIVE']}",
        f"  BAD  -> PASA    (CRITICO)     : {counts['BAD_PASS_CRITICAL_FALSE_ACCEPT']}", "",
        "Metricas de negocio:",
        f"  false_reject_rate  (GOOD->REC/GOOD) : {false_reject_rate:.1%}",
        f"  false_accept_rate  (BAD->PASA/BAD)  : {false_accept_rate:.1%}",
        f"  automatic_accept   (PASA/total)      : {auto_accept_rate:.1%}",
        f"  manual_review_rate (REVISAR/total)   : {manual_review_rate:.1%}",
        f"  reject_rate        (RECHAZA/total)   : {reject_rate:.1%}", "",
        "Archivos:",
        f"  {OUT_DIR / 'results_final_u3_bad_regression.csv'}",
        f"  {OUT_DIR / 'contact_sheet_all.jpg'}",
        f"  {OUT_DIR / 'contact_sheet_bad_pass_critical.jpg'}",
        f"  {OUT_DIR / 'contact_sheet_good_reject_critical.jpg'}",
        f"  {OUT_DIR / 'contact_sheet_review_cases.jpg'}",
    ]
    (OUT_DIR / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    print()
    for line in summary_lines:
        print(line)

    # --- Reporte Markdown ---
    # Determinar conclusion
    if counts["BAD_PASS_CRITICAL_FALSE_ACCEPT"] == 0 and counts["GOOD_REJECT_CRITICAL_FALSE_REJECT"] == 0:
        conclusion = "ACEPTAR (sin errores criticos en ninguna direccion)"
        accept_final = "SI"
    elif false_accept_rate <= 0.05 and false_reject_rate == 0.0:
        conclusion = "ACEPTAR CON MONITOREO (FAR <= 5%, FRR = 0%)"
        accept_final = "SI"
    elif false_accept_rate <= 0.10 and false_reject_rate == 0.0:
        conclusion = "ACEPTAR PROVISIONAL (FAR <= 10%, FRR = 0%, reforzar detector de defectos)"
        accept_final = "SI provisional"
    elif false_accept_rate > 0.10:
        conclusion = "NO ACEPTAR todavia: FAR > 10%, reforzar strong_defect_evidence/detector"
        accept_final = "NO"
    else:
        conclusion = "REVISAR: FRR > 0 o FAR ambiguo"
        accept_final = "CONDICIONAL"

    # Lista de falsos aceptados para el reporte
    bad_pass_list = "\n".join(
        f"| {r['image_name']} | {r['u3_p_good']:.3f} | {r['u3_p_bad']:.3f} | {r['u3_raw']} | {r['reason']} |"
        for r in results if r['business_result'] == 'BAD_PASS_CRITICAL_FALSE_ACCEPT'
    ) or "_ninguno_"

    good_reject_list = "\n".join(
        f"| {r['image_name']} | {r['u3_p_good']:.3f} | {r['u3_p_bad']:.3f} | {r['reason']} |"
        for r in results if r['business_result'] == 'GOOD_REJECT_CRITICAL_FALSE_REJECT'
    ) or "_ninguno_"

    report_md = f"""# Reporte: Validación de Regresión BAD — Pipeline U3 Fusion v2

**Fecha:** 2026-05-21
**Dataset:** quality_fruits360_human_v1 — {total} imágenes etiquetadas por humano

---

## Resultados

| Categoría | N | % |
|---|---|---|
| GOOD → PASA (OK) | {counts['GOOD_PASS_OK']} | {counts['GOOD_PASS_OK']/good_total:.1%} de GOOD |
| GOOD → REVISAR (conservador) | {counts['GOOD_REVIEW_OK_CONSERVATIVE']} | {counts['GOOD_REVIEW_OK_CONSERVATIVE']/good_total:.1%} de GOOD |
| GOOD → RECHAZA (**CRITICO**) | {counts['GOOD_REJECT_CRITICAL_FALSE_REJECT']} | {false_reject_rate:.1%} de GOOD |
| BAD → RECHAZA (OK) | {counts['BAD_REJECT_OK']} | {counts['BAD_REJECT_OK']/bad_total:.1%} de BAD |
| BAD → REVISAR (conservador) | {counts['BAD_REVIEW_OK_CONSERVATIVE']} | {counts['BAD_REVIEW_OK_CONSERVATIVE']/bad_total:.1%} de BAD |
| BAD → PASA (**CRITICO**) | {counts['BAD_PASS_CRITICAL_FALSE_ACCEPT']} | {false_accept_rate:.1%} de BAD |

## Métricas de negocio

| Métrica | Valor |
|---|---|
| false_reject_rate (GOOD→RECHAZA/GOOD) | {false_reject_rate:.1%} |
| false_accept_rate (BAD→PASA/BAD) | {false_accept_rate:.1%} |
| automatic_accept_rate (PASA/total) | {auto_accept_rate:.1%} |
| manual_review_rate (REVISAR/total) | {manual_review_rate:.1%} |
| reject_rate (RECHAZA/total) | {reject_rate:.1%} |

---

## Falsos aceptados — BAD → PASA ({counts['BAD_PASS_CRITICAL_FALSE_ACCEPT']})

| Imagen | p_good | p_bad | U3_raw | Razón |
|---|---|---|---|---|
{bad_pass_list}

## Falsos rechazos — GOOD → RECHAZA ({counts['GOOD_REJECT_CRITICAL_FALSE_REJECT']})

| Imagen | p_good | p_bad | Razón |
|---|---|---|---|
{good_reject_list}

---

## Interpretación (TAREA 6)

### 1. ¿El fix U3 ha eliminado falsos rechazos de peras sanas?

**Resultado anterior en supermercado:** 86/86 PASA (0 RECHAZA, 0 REVISAR).
**Resultado en dataset humano GOOD ({good_total} imgs):** {counts['GOOD_PASS_OK']} PASA + {counts['GOOD_REVIEW_OK_CONSERVATIVE']} REVISAR + {counts['GOOD_REJECT_CRITICAL_FALSE_REJECT']} RECHAZA.
FRR = {false_reject_rate:.1%}

{"→ Sí, los falsos rechazos de GOOD son mínimos o nulos." if counts['GOOD_REJECT_CRITICAL_FALSE_REJECT'] == 0 else f"→ Hay {counts['GOOD_REJECT_CRITICAL_FALSE_REJECT']} falsos rechazos de GOOD que requieren análisis."}

### 2. ¿Aparecen falsos aceptados BAD→PASA?

{f"→ Sí: {counts['BAD_PASS_CRITICAL_FALSE_ACCEPT']} BAD peras son incorrectamente aceptadas como PASA ({false_accept_rate:.1%})." if counts['BAD_PASS_CRITICAL_FALSE_ACCEPT'] > 0 else "→ No hay falsos aceptados en este dataset."}

### 3. ¿Está demasiado permisivo el pipeline?

{"→ Sí. La tasa de aceptación automática es " + f"{auto_accept_rate:.1%}, con {false_accept_rate:.1%} de falsos aceptados BAD. Reforzar `strong_defect_evidence` con detector YOLO de defectos." if false_accept_rate > 0.10 else "→ La permisividad está dentro de rango tolerable para un prototipo académico."}

### 4. ¿Está demasiado conservador?

{"→ Sí. La tasa de revisión manual es " + f"{manual_review_rate:.1%}, lo que significa que {total_revisar} imágenes quedan pendientes de revisión humana." if manual_review_rate > 0.5 else f"→ La tasa de revisión manual es {manual_review_rate:.1%}, razonable para la configuración actual sin YOLO de defectos activo."}

### 5. ¿Se puede integrar U3 como versión final provisional?

**{accept_final}** — {conclusion}

**Observación clave:** El modo `U3_BAD + no strong_defect → REVISAR` (en lugar de RECHAZA directo) hace que la mayoría de peras BAD correctamente identificadas por U3 queden en REVISAR en vez de RECHAZA. Esto es conservador pero correcto para un prototipo sin detector de defectos confirmado. Para producción real se necesitaría activar el YOLO de defectos (`--use-defect-model`) o reducir el umbral de `strong_defect_evidence`.

---

## Archivos generados

- `outputs/final_u3_bad_regression_eval/results_final_u3_bad_regression.csv`
- `outputs/final_u3_bad_regression_eval/contact_sheet_all.jpg`
- `outputs/final_u3_bad_regression_eval/contact_sheet_bad_pass_critical.jpg`
- `outputs/final_u3_bad_regression_eval/contact_sheet_good_reject_critical.jpg`
- `outputs/final_u3_bad_regression_eval/contact_sheet_review_cases.jpg`
- `outputs/final_u3_bad_regression_eval/summary.txt`
"""
    (REPORTS_DIR / "final_u3_bad_regression_report.md").write_text(report_md, encoding="utf-8")
    print(f"\nReporte: {REPORTS_DIR / 'final_u3_bad_regression_report.md'}")

    return counts, false_reject_rate, false_accept_rate


if __name__ == "__main__":
    main()
