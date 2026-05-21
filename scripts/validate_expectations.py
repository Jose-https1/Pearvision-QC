"""
validate_expectations.py — Valida resultados del pipeline contra verdad humana.

Lee:
  outputs/quality_analysis/resultados_calidad.csv   (resultados del pipeline)
  data/samples_quality_controlled_test_expectations.csv  (verdad humana)

Imprime un informe de PASS / FAIL por imagen y un resumen final.
Genera: outputs/quality_audit/validation_report.txt
"""

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_CSV = PROJECT_ROOT / "outputs" / "quality_analysis" / "resultados_calidad.csv"
EXPECT_CSV  = PROJECT_ROOT / "data" / "samples_quality_controlled_test_expectations.csv"
REPORT_PATH = PROJECT_ROOT / "outputs" / "quality_audit" / "validation_report.txt"


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return {row["image"]: row for row in csv.DictReader(f)}


def validate(results, expectations):
    lines = []
    passed = []
    failed = []
    not_found = []

    for img, exp in expectations.items():
        allowed = set(exp["allowed_decisions"].split("|"))
        group   = exp["expected_group"]
        notes   = exp.get("notes", "")

        if img not in results:
            not_found.append(img)
            lines.append(f"  NOT_FOUND  {img:<30} (no está en resultados_calidad.csv)")
            continue

        decision = results[img]["decision"]
        defect   = results[img].get("defect_pct", "?")
        rot      = results[img].get("dark_rot_pct", "?")
        maxr     = results[img].get("max_region_pct", "?")
        l_mean   = results[img].get("body_l_mean", "?")

        ok = decision in allowed
        tag = "PASS" if ok else "FAIL"
        if ok:
            passed.append(img)
        else:
            failed.append(img)

        lines.append(
            f"  {tag:<6} [{group:<18}] {img:<30} "
            f"decision={decision:<8} allowed={exp['allowed_decisions']:<14} "
            f"def={defect}% rot={rot}% max={maxr}% L={l_mean}"
        )
        if not ok:
            lines.append(f"         >> NOTA: {notes}")

    sep = "=" * 90
    header = [
        sep,
        "PearVision QC — Validacion contra verdad humana",
        sep,
    ]
    footer = [
        "",
        sep,
        f"  TOTAL EXPECTATIVAS : {len(expectations)}",
        f"  PASS               : {len(passed)}",
        f"  FAIL               : {len(failed)}",
        f"  NOT FOUND          : {len(not_found)}",
        "",
    ]
    if failed:
        footer.append("  IMAGENES CON FALLO:")
        for f in failed:
            footer.append(f"    - {f}")
    else:
        footer.append("  TODAS LAS EXPECTATIVAS CUMPLIDAS.")
    footer.append(sep)

    all_lines = header + [""] + lines + footer
    return all_lines, len(failed) == 0 and len(not_found) == 0


def main():
    if not RESULTS_CSV.exists():
        print(f"ERROR: {RESULTS_CSV} no existe. Ejecuta analyze_quality.py --save primero.")
        sys.exit(1)
    if not EXPECT_CSV.exists():
        print(f"ERROR: {EXPECT_CSV} no existe.")
        sys.exit(1)

    results      = load_csv(RESULTS_CSV)
    expectations = load_csv(EXPECT_CSV)

    lines, all_ok = validate(results, expectations)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    for line in lines:
        print(line)

    if all_ok:
        print("\n  RESULTADO FINAL: TODAS LAS EXPECTATIVAS CUMPLIDAS.")
        sys.exit(0)
    else:
        print("\n  RESULTADO FINAL: HAY FALLOS. Revisar umbrales o logica.")
        sys.exit(1)


if __name__ == "__main__":
    main()
