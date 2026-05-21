"""
validate_fruits360_good_eval.py
Valida los resultados del pipeline sobre el dataset Fruits-360 good pears.

Lee:
  data/samples_quality_fruits360_good_eval_expectations.csv
  outputs/quality_analysis_fruits360_good_eval/resultados_calidad.csv

Genera:
  outputs/quality_audit_fruits360_good_eval/validation_report.txt
"""

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXPECT_CSV = PROJECT_ROOT / "data" / "samples_quality_fruits360_good_eval_expectations.csv"
RESULTS_CSV = PROJECT_ROOT / "outputs" / "quality_analysis_fruits360_good_eval" / "resultados_calidad.csv"
AUDIT_DIR   = PROJECT_ROOT / "outputs" / "quality_audit_fruits360_good_eval"
REPORT_PATH = AUDIT_DIR / "validation_report.txt"


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return {row["image"]: row for row in csv.DictReader(f)}


def main():
    if not EXPECT_CSV.exists():
        print(f"ERROR: {EXPECT_CSV} no existe.")
        sys.exit(1)
    if not RESULTS_CSV.exists():
        print(f"ERROR: {RESULTS_CSV} no existe. Ejecuta analyze_quality.py primero.")
        sys.exit(1)

    expectations = load_csv(EXPECT_CSV)
    results = load_csv(RESULTS_CSV)

    total  = len(expectations)
    pasa_list    = []
    revisar_list = []
    rechaza_list = []
    not_found    = []

    lines = []

    for img, exp in sorted(expectations.items()):
        allowed = set(exp["allowed_decisions"].split("|"))
        if img not in results:
            not_found.append(img)
            lines.append(f"  NOT_FOUND  {img}")
            continue

        r        = results[img]
        decision = r["decision"]
        defect   = r.get("defect_pct", "?")
        rot      = r.get("dark_rot_pct", "?")
        maxr     = r.get("max_region_pct", "?")
        brown    = r.get("brown_dark_pct", "?")
        l_mean   = r.get("body_l_mean", "?")
        cls_pred = r.get("quality_cls_pred", "?")
        cls_bad  = r.get("quality_cls_bad_conf", "?")
        yolo_v   = r.get("yolo_defect_count", "0")
        label    = r.get("display_label", decision)
        cap_ok   = r.get("capture_valid", "true").lower() in ("true", "1")

        ok = decision in allowed
        if decision == "PASA":
            pasa_list.append(img)
        elif decision == "REVISAR":
            revisar_list.append(img)
        else:
            rechaza_list.append(img)

        tag = "PASS" if ok else "FAIL"
        cap_tag = "" if cap_ok else " [cap_invalid]"
        lines.append(
            f"  {tag:<4} {decision:<8} {img:<45} "
            f"def={defect}% rot={rot}% max={maxr}% "
            f"brown={brown}% L={l_mean} "
            f"cls={cls_pred}/{cls_bad} yolo={yolo_v}{cap_tag}"
        )
        if not ok:
            lines.append(f"       >> FALLO: esperado={exp['allowed_decisions']}  obtenido={decision}")
            lines.append(f"          label={label}")

    cap_invalid = sum(1 for img in (pasa_list + revisar_list + rechaza_list)
                      if results.get(img, {}).get("capture_valid", "true").lower() not in ("true", "1"))

    false_reject_pct = len(rechaza_list) / max(1, total) * 100
    status = "OK" if false_reject_pct < 5.0 else "NEEDS_RULE_FIX"

    sep = "=" * 100
    header = [
        sep,
        "PearVision QC -- Fruits-360 Good Eval -- Validacion contra verdad humana",
        sep,
        "",
    ]
    footer = [
        "",
        sep,
        f"  TOTAL EXPECTATIVAS   : {total}",
        f"  NOT FOUND            : {len(not_found)}",
        f"  PASA                 : {len(pasa_list)}",
        f"  REVISAR              : {len(revisar_list)}  (captura invalida: {cap_invalid})",
        f"  RECHAZA              : {len(rechaza_list)}  <-- falsos rechazos",
        f"  FALSE_REJECT_RATE    : {false_reject_pct:.1f}%",
        f"  STATUS               : {status}",
        "",
    ]

    if rechaza_list:
        footer.append("  FALSOS RECHAZOS:")
        for img in rechaza_list:
            r = results.get(img, {})
            footer.append(
                f"    - {img:<45} "
                f"def={r.get('defect_pct','?')}% "
                f"rot={r.get('dark_rot_pct','?')}% "
                f"max={r.get('max_region_pct','?')}%"
            )

    footer.append(sep)

    # Análisis de causa de RECHAZA
    if rechaza_list:
        footer += ["", "  ANALISIS DE CAUSA (falsos rechazos):"]
        high_def  = [img for img in rechaza_list if float(results[img].get("defect_pct","0") or 0) >= 40.0]
        high_rot  = [img for img in rechaza_list if float(results[img].get("dark_rot_pct","0") or 0) >= 20.0]
        high_max  = [img for img in rechaza_list if float(results[img].get("max_region_pct","0") or 0) >= 25.0]
        footer.append(f"    defect_pct >= 40%  : {len(high_def)} imagenes")
        footer.append(f"    dark_rot_pct >= 20%: {len(high_rot)} imagenes")
        footer.append(f"    max_region >= 25%  : {len(high_max)} imagenes")

    all_lines = header + lines + footer
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(all_lines), encoding="utf-8")

    for line in all_lines:
        print(line)

    print(f"\n  Reporte guardado: {REPORT_PATH}")
    sys.exit(0 if false_reject_pct < 5.0 else 1)


if __name__ == "__main__":
    main()
