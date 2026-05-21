"""
apply_error_corrections_v2.py

Aplica las correcciones humanas de los errores del clasificador V1:
  F360_0198: BAD   -> REVIEW
  F360_0052: GOOD  -> REVIEW
  F360_0060: GOOD  -> GOOD  (sin cambio, solo confirmar)

Modifica:  data/fruits360_human_review/human_labels_template.csv
Genera:    reports/fruits360_error_corrections_v2_report.md

No entrena ningun modelo.
No modifica analyze_quality.py ni quality_rules.yaml.
"""

import csv
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_CSV = PROJECT_ROOT / "data" / "fruits360_human_review" / "human_labels_template.csv"
REPORT_MD    = PROJECT_ROOT / "reports" / "fruits360_error_corrections_v2_report.md"

CORRECTIONS = {
    "F360_0198": ("BAD",  "REVIEW"),
    "F360_0052": ("GOOD", "REVIEW"),
    "F360_0060": ("GOOD", "GOOD"),   # confirmacion, sin cambio real
}


def main():
    print("=== apply_error_corrections_v2 ===")
    if not TEMPLATE_CSV.exists():
        print(f"ERROR: {TEMPLATE_CSV} no existe.")
        sys.exit(1)

    with TEMPLATE_CSV.open(encoding="utf-8", newline="") as f:
        reader    = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows       = list(reader)

    row_map = {r["review_id"]: r for r in rows}

    # Verificar estado previo
    print("  Estado ANTES de correcciones:")
    issues = []
    for rid, (expected_old, new_lbl) in CORRECTIONS.items():
        if rid not in row_map:
            issues.append(f"  {rid}: NO ENCONTRADO en CSV")
            continue
        actual = row_map[rid]["human_label"]
        match  = "OK" if actual == expected_old else f"ADVERTENCIA (esperado={expected_old})"
        print(f"    {rid}: actual={actual}  esperado={expected_old}  -> nuevo={new_lbl}  [{match}]")
        if actual != expected_old:
            issues.append(f"{rid}: esperado={expected_old} pero actual={actual}")

    if issues:
        print("  ADVERTENCIAS detectadas — se aplican de todas formas:")
        for iss in issues:
            print(f"    {iss}")

    # Aplicar correcciones
    applied = []
    skipped = []
    for rid, (expected_old, new_lbl) in CORRECTIONS.items():
        if rid not in row_map:
            continue
        old = row_map[rid]["human_label"]
        if old == new_lbl:
            skipped.append((rid, old, new_lbl))
        else:
            row_map[rid]["human_label"] = new_lbl
            applied.append((rid, old, new_lbl))

    print(f"\n  Cambios aplicados: {len(applied)}")
    for rid, old, new in applied:
        print(f"    {rid}: {old} -> {new}")
    print(f"  Sin cambio (ya correctos): {len(skipped)}")
    for rid, old, new in skipped:
        print(f"    {rid}: {old} (sin cambio)")

    # Guardar CSV
    with TEMPLATE_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  CSV guardado: {TEMPLATE_CSV.name}")

    # Verificar estado post
    with TEMPLATE_CSV.open(encoding="utf-8", newline="") as f:
        rows_post = list(csv.DictReader(f))
    row_map_post = {r["review_id"]: r for r in rows_post}

    print("  Estado DESPUES de correcciones:")
    all_ok = True
    for rid, (_, new_lbl) in CORRECTIONS.items():
        actual = row_map_post[rid]["human_label"]
        ok     = actual == new_lbl
        if not ok:
            all_ok = False
        print(f"    {rid}: {actual}  {'OK' if ok else 'FALLO'}")

    # Conteos globales post-correccion
    counts = {}
    for r in rows_post:
        lbl = r["human_label"]
        if lbl:
            counts[lbl] = counts.get(lbl, 0) + 1

    print(f"\n  Conteos globales post-correccion:")
    for lbl in ["GOOD", "BAD", "REVIEW", "INVALID"]:
        print(f"    {lbl:<8}: {counts.get(lbl, 0)}")
    print(f"    TOTAL   : {sum(counts.values())}")

    status = "PASS" if all_ok else "FAIL"
    print(f"\n  STATUS: {status}")

    # Reporte
    lines = [
        "# Correcciones de Errores V2 — Fruits-360",
        "",
        f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Status: **{status}**",
        "",
        "## Correcciones aplicadas",
        "",
        "| review_id | Etiqueta anterior | Etiqueta nueva | Accion |",
        "|-----------|-------------------|----------------|--------|",
    ]
    for rid, old, new in applied:
        lines.append(f"| {rid} | {old} | {new} | CAMBIADO |")
    for rid, old, new in skipped:
        lines.append(f"| {rid} | {old} | {new} | SIN CAMBIO (confirmado) |")

    lines += [
        "",
        "## Conteos globales post-correccion",
        "",
        "| Etiqueta | Count |",
        "|----------|-------|",
    ]
    for lbl in ["GOOD", "BAD", "REVIEW", "INVALID"]:
        lines.append(f"| {lbl} | {counts.get(lbl, 0)} |")
    lines.append(f"| **TOTAL** | **{sum(counts.values())}** |")

    lines += [
        "",
        "## Motivacion de cada correccion",
        "",
        "- **F360_0198** (BAD->REVIEW): clasificador V1 la predijo como GOOD con confianza 0.995.",
        "  Inspeccion visual confirma que es un caso ambiguo — se mueve a REVIEW.",
        "- **F360_0052** (GOOD->REVIEW): clasificador V1 la predijo como BAD con confianza 0.665.",
        "  Inspeccion visual confirma ambiguedad — se mueve a REVIEW.",
        "- **F360_0060** (GOOD->GOOD): clasificador V1 la predijo como BAD con confianza 0.957.",
        "  Inspeccion visual confirma que SI es GOOD — etiqueta correcta, modelo fallaba.",
        "",
        "## Confirmaciones",
        "",
        "- analyze_quality.py NO fue modificado.",
        "- quality_rules.yaml NO fue modificado.",
        "- NO se entrenó ningun modelo.",
    ]

    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Reporte: {REPORT_MD.name}")

    return status


if __name__ == "__main__":
    main()
