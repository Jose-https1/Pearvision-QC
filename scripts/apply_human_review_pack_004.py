"""
apply_human_review_pack_004.py
Aplica las etiquetas humanas del PACK 004 (F360_0091–F360_0120).

Modifica:  data/fruits360_human_review/human_labels_template.csv
Genera:    reports/fruits360_human_review_pack004_apply_report.md
"""

import csv
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEMPLATE_CSV = PROJECT_ROOT / "data" / "fruits360_human_review" / "human_labels_template.csv"
MASTER_CSV   = PROJECT_ROOT / "data" / "fruits360_human_review" / "fruits360_human_review_master.csv"
REPORT_MD    = PROJECT_ROOT / "reports" / "fruits360_human_review_pack004_apply_report.md"

PACK_NUM      = 4
PACK_ID_START = 91
PACK_ID_END   = 120
VALID_LABELS  = {"GOOD", "BAD", "INVALID", "REVIEW"}

LABELS = {
    "F360_0091": "BAD",
    "F360_0092": "BAD",
    "F360_0093": "BAD",
    "F360_0094": "REVIEW",
    "F360_0095": "GOOD",
    "F360_0096": "BAD",
    "F360_0097": "BAD",
    "F360_0098": "BAD",
    "F360_0099": "GOOD",
    "F360_0100": "REVIEW",
    "F360_0101": "REVIEW",
    "F360_0102": "BAD",
    "F360_0103": "REVIEW",
    "F360_0104": "REVIEW",
    "F360_0105": "BAD",
    "F360_0106": "BAD",
    "F360_0107": "BAD",
    "F360_0108": "REVIEW",
    "F360_0109": "BAD",
    "F360_0110": "GOOD",
    "F360_0111": "BAD",
    "F360_0112": "BAD",
    "F360_0113": "REVIEW",
    "F360_0114": "BAD",
    "F360_0115": "BAD",
    "F360_0116": "BAD",
    "F360_0117": "BAD",
    "F360_0118": "BAD",
    "F360_0119": "BAD",
    "F360_0120": "REVIEW",
}

EXPECTED      = {"GOOD": 3, "BAD": 19, "INVALID": 0, "REVIEW": 8}
PROTECTED     = {f"F360_{i:04d}" for i in range(1, 91)}   # packs 001-003

# Conteos esperados de packs protegidos (para verificación)
PROTECTED_EXPECTED = {
    "001": {"GOOD": 7,  "BAD": 19, "REVIEW": 4},
    "002": {"GOOD": 9,  "BAD": 17, "REVIEW": 4},
    "003": {"GOOD": 8,  "BAD": 17, "REVIEW": 5},
}


def validate(labels: dict, rows: list) -> list[str]:
    issues = []
    if len(labels) != 30:
        issues.append(f"FALLO: se esperaban 30 IDs, hay {len(labels)}")

    csv_ids = {r["review_id"] for r in rows}
    missing = [rid for rid in labels if rid not in csv_ids]
    if missing:
        issues.append(f"FALLO: IDs no encontrados en CSV: {missing}")

    out_of_range = [rid for rid in labels
                    if not (PACK_ID_START <= int(rid.split("_")[1]) <= PACK_ID_END)]
    if out_of_range:
        issues.append(f"FALLO: IDs fuera de rango {PACK_ID_START}-{PACK_ID_END}: {out_of_range}")

    overlap = PROTECTED & set(labels.keys())
    if overlap:
        issues.append(f"FALLO: solapamiento con packs anteriores: {overlap}")

    counts = {}
    for lbl in labels.values():
        counts[lbl] = counts.get(lbl, 0) + 1
    for lbl, exp in EXPECTED.items():
        got = counts.get(lbl, 0)
        if got != exp:
            issues.append(f"FALLO: {lbl} esperado={exp} obtenido={got}")

    return issues


def verify_protected_packs(rows: list) -> list[str]:
    issues = []
    ranges = {"001": range(1, 31), "002": range(31, 61), "003": range(61, 91)}
    for pack, rng in ranges.items():
        pack_ids = {f"F360_{i:04d}" for i in rng}
        pack_rows = [r for r in rows if r["review_id"] in pack_ids]
        counts = {}
        for r in pack_rows:
            lbl = r["human_label"]
            if lbl:
                counts[lbl] = counts.get(lbl, 0) + 1
        exp = PROTECTED_EXPECTED[pack]
        for lbl, n in exp.items():
            got = counts.get(lbl, 0)
            if got != n:
                issues.append(f"FALLO: PACK {pack} {lbl} esperado={n} obtenido={got}")
    return issues


def apply_labels(labels: dict) -> tuple[list, int]:
    with open(TEMPLATE_CSV, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    updated = 0
    for row in rows:
        if row["review_id"] in labels:
            row["human_label"] = labels[row["review_id"]]
            updated += 1

    with open(TEMPLATE_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return rows, updated


def write_report(labels: dict, updated: int, issues: list, prot_issues: list, rows: list) -> str:
    by_label: dict[str, list] = {}
    for rid, lbl in sorted(labels.items()):
        by_label.setdefault(lbl, []).append(rid)

    total_labeled = sum(1 for r in rows if r["human_label"].strip())
    all_issues = issues + prot_issues
    status = "PASS" if not all_issues else "FAIL"

    lines = [
        "# Fruits-360 Human Review — Aplicación PACK 004",
        "",
        f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Pack aplicado: PACK 004 (IDs F360_0091 – F360_0120)",
        f"Status: **{status}**",
        "",
        "## Conteos finales PACK 004",
        "",
        "| Etiqueta | Esperado | Obtenido | IDs |",
        "|----------|----------|----------|-----|",
    ]
    for lbl in ["GOOD", "BAD", "INVALID", "REVIEW"]:
        ids = by_label.get(lbl, [])
        exp = EXPECTED.get(lbl, 0)
        mark = "OK" if len(ids) == exp else "FALLO"
        id_str = ", ".join(ids) if ids else "(ninguno)"
        lines.append(f"| {lbl} | {exp} | {len(ids)} {mark} | {id_str} |")

    lines += [
        "",
        "## Estado de packs protegidos",
        "",
        "| Pack | GOOD | BAD | REVIEW | Estado |",
        "|------|------|-----|--------|--------|",
    ]
    ranges = {"001": range(1, 31), "002": range(31, 61), "003": range(61, 91)}
    for pack, rng in ranges.items():
        pack_ids = {f"F360_{i:04d}" for i in rng}
        counts = {}
        for r in rows:
            if r["review_id"] in pack_ids and r["human_label"]:
                counts[r["human_label"]] = counts.get(r["human_label"], 0) + 1
        exp = PROTECTED_EXPECTED[pack]
        ok = all(counts.get(k, 0) == v for k, v in exp.items())
        lines.append(f"| {pack} | {counts.get('GOOD',0)} | {counts.get('BAD',0)} | "
                     f"{counts.get('REVIEW',0)} | {'OK' if ok else 'FALLO'} |")

    lines += [
        "",
        "## Resumen global",
        f"- Total etiquetadas: {total_labeled}/300",
        f"- Packs completados: 001, 002, 003, 004 (120/300)",
        "",
        "## Confirmaciones",
        "- NO se entrenó ningún modelo.",
        "- Solo se actualizó la columna `human_label` para los 30 IDs del PACK 004.",
        "- PACK 001, 002 y 003 no fueron modificados.",
    ]
    if all_issues:
        lines += ["", "## Errores", ""]
        for iss in all_issues:
            lines.append(f"- {iss}")

    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Reporte: {REPORT_MD}")
    return status


def main():
    print("=== apply_human_review_pack_004 ===")

    for path in [TEMPLATE_CSV, MASTER_CSV]:
        if not path.exists():
            print(f"ERROR: {path} no existe.")
            sys.exit(1)

    with open(TEMPLATE_CSV, encoding="utf-8", newline="") as f:
        rows_pre = list(csv.DictReader(f))

    print("  Validando consistencia...")
    issues = validate(LABELS, rows_pre)
    if issues:
        for iss in issues:
            print(f"    {iss}")
        sys.exit(1)
    print("  Validacion OK")

    print("  Aplicando etiquetas...")
    rows, updated = apply_labels(LABELS)
    print(f"  Actualizadas: {updated} filas")

    print("  Verificando packs protegidos (001, 002, 003)...")
    prot_issues = verify_protected_packs(rows)
    if prot_issues:
        for iss in prot_issues:
            print(f"    {iss}")
    else:
        print("  Packs 001/002/003 intactos")

    status = write_report(LABELS, updated, issues, prot_issues, rows)

    counts = {}
    for row in rows:
        if row["review_id"] in LABELS:
            counts[row["human_label"]] = counts.get(row["human_label"], 0) + 1

    total_labeled = sum(1 for r in rows if r["human_label"].strip())

    print()
    print("=== RESUMEN PACK 004 ===")
    for lbl in ["GOOD", "BAD", "INVALID", "REVIEW"]:
        print(f"  {lbl:<8}: {counts.get(lbl, 0)}")
    print(f"  TOTAL   : {updated}")
    print(f"  STATUS  : {status}")
    print(f"\n  CSV total etiquetado: {total_labeled}/300")
    print(f"  Siguiente paso: revisar review_pack_005.jpg y aplicar PACK 005")


if __name__ == "__main__":
    main()
