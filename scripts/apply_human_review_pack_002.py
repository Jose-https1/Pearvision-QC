"""
apply_human_review_pack_002.py
Aplica las etiquetas humanas del PACK 002 al CSV de revisión de Fruits-360.

Modifica:  data/fruits360_human_review/human_labels_template.csv
Genera:    reports/fruits360_human_review_pack002_apply_report.md
"""

import csv
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEMPLATE_CSV = PROJECT_ROOT / "data" / "fruits360_human_review" / "human_labels_template.csv"
MASTER_CSV   = PROJECT_ROOT / "data" / "fruits360_human_review" / "fruits360_human_review_master.csv"
REPORT_MD    = PROJECT_ROOT / "reports" / "fruits360_human_review_pack002_apply_report.md"

PACK_NUM      = 2
IMGS_PER_PACK = 30
PACK_ID_START = (PACK_NUM - 1) * IMGS_PER_PACK + 1   # 31
PACK_ID_END   = PACK_NUM * IMGS_PER_PACK               # 60

# ---------------------------------------------------------------------------
# Etiquetas humanas oficiales PACK 002
# ---------------------------------------------------------------------------
LABELS = {
    "F360_0031": "GOOD",
    "F360_0032": "BAD",
    "F360_0033": "GOOD",
    "F360_0034": "BAD",
    "F360_0035": "BAD",
    "F360_0036": "GOOD",
    "F360_0037": "REVIEW",
    "F360_0038": "BAD",
    "F360_0039": "BAD",
    "F360_0040": "GOOD",
    "F360_0041": "BAD",
    "F360_0042": "REVIEW",
    "F360_0043": "BAD",
    "F360_0044": "GOOD",
    "F360_0045": "GOOD",
    "F360_0046": "BAD",
    "F360_0047": "BAD",
    "F360_0048": "GOOD",
    "F360_0049": "BAD",
    "F360_0050": "BAD",
    "F360_0051": "BAD",
    "F360_0052": "GOOD",
    "F360_0053": "BAD",
    "F360_0054": "BAD",
    "F360_0055": "REVIEW",
    "F360_0056": "BAD",
    "F360_0057": "REVIEW",
    "F360_0058": "BAD",
    "F360_0059": "BAD",
    "F360_0060": "GOOD",
}

VALID_LABELS = {"GOOD", "BAD", "INVALID", "REVIEW"}


# ---------------------------------------------------------------------------
# TAREA 3 — Validación de consistencia
# ---------------------------------------------------------------------------

def validate(labels: dict, master_rows: list) -> list:
    issues = []

    # 1. Exactamente 30 IDs únicos
    if len(labels) != 30:
        issues.append(f"FALLO: se esperaban 30 IDs únicos, hay {len(labels)}")

    # 2. Sin IDs repetidos entre categorías (garantizado por dict, pero verificar)
    all_ids = list(labels.keys())
    if len(all_ids) != len(set(all_ids)):
        issues.append("FALLO: hay IDs repetidos entre categorías.")

    # 3. Todos los IDs existen en el master
    master_ids = {r["review_id"] for r in master_rows}
    missing_in_master = [rid for rid in labels if rid not in master_ids]
    if missing_in_master:
        issues.append(f"FALLO: IDs no encontrados en master: {missing_in_master}")

    # 4. Todos los IDs pertenecen al pack 002 (rango 31–60)
    out_of_range = []
    for rid in labels:
        try:
            num = int(rid.split("_")[1])
            if not (PACK_ID_START <= num <= PACK_ID_END):
                out_of_range.append(rid)
        except Exception:
            out_of_range.append(rid)
    if out_of_range:
        issues.append(f"FALLO: IDs fuera del rango del pack 002 ({PACK_ID_START}-{PACK_ID_END}): {out_of_range}")

    # 5. Etiquetas válidas
    invalid_labels = {rid: lbl for rid, lbl in labels.items() if lbl not in VALID_LABELS}
    if invalid_labels:
        issues.append(f"FALLO: etiquetas no válidas: {invalid_labels}")

    # 6. Conteos esperados
    counts = {}
    for lbl in labels.values():
        counts[lbl] = counts.get(lbl, 0) + 1
    expected = {"GOOD": 9, "BAD": 17, "INVALID": 0, "REVIEW": 4}
    for lbl, exp_n in expected.items():
        got_n = counts.get(lbl, 0)
        if got_n != exp_n:
            issues.append(f"FALLO: {lbl} esperado={exp_n} obtenido={got_n}")

    return issues


# ---------------------------------------------------------------------------
# TAREA 2 — Aplicar etiquetas
# ---------------------------------------------------------------------------

def apply_labels(labels: dict) -> tuple[list, int]:
    with open(TEMPLATE_CSV, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
        fieldnames = rows[0].keys() if rows else []

    # Re-leer para obtener fieldnames limpios
    with open(TEMPLATE_CSV, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    updated = 0
    for row in rows:
        rid = row["review_id"]
        if rid in labels:
            row["human_label"] = labels[rid]
            updated += 1

    with open(TEMPLATE_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return rows, updated


# ---------------------------------------------------------------------------
# TAREA 4 — Reporte
# ---------------------------------------------------------------------------

def write_report(labels: dict, updated: int, issues: list):
    by_label: dict[str, list] = {}
    for rid, lbl in sorted(labels.items()):
        by_label.setdefault(lbl, []).append(rid)

    status = "PASS" if not issues else "FAIL"
    lines = [
        "# Fruits-360 Human Review — Aplicación PACK 002",
        "",
        f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Pack aplicado: PACK 002 (IDs F360_0031 – F360_0060)",
        f"Status: **{status}**",
        "",
        "## Conteos por etiqueta",
        "",
        "| Etiqueta | Cantidad | IDs |",
        "|----------|----------|-----|",
    ]
    for lbl in ["GOOD", "BAD", "INVALID", "REVIEW"]:
        ids = by_label.get(lbl, [])
        id_str = ", ".join(ids) if ids else "(ninguno)"
        lines.append(f"| {lbl} | {len(ids)} | {id_str} |")

    lines += [
        "",
        "## Validaciones",
        "",
    ]
    if not issues:
        lines.append("- OK: 30 IDs únicos en PACK 002.")
        lines.append("- OK: Sin IDs repetidos entre categorías.")
        lines.append("- OK: Todos los IDs existen en el master.")
        lines.append("- OK: Todos los IDs pertenecen al rango 31–60 (PACK 002).")
        lines.append("- OK: Conteos: GOOD=9, BAD=17, INVALID=0, REVIEW=4.")
    else:
        for issue in issues:
            lines.append(f"- {issue}")

    lines += [
        "",
        "## Archivos modificados",
        "",
        f"- `data/fruits360_human_review/human_labels_template.csv` — {updated} filas actualizadas",
        "",
        "## Confirmaciones",
        "",
        "- NO se entrenó ningún modelo.",
        "- Solo se actualizó el CSV de revisión humana.",
        "- El master CSV NO fue modificado.",
        "- Los IDs, imágenes y rutas originales no fueron alterados.",
    ]

    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Reporte: {REPORT_MD}")
    return status


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"=== apply_human_review_pack_002 ===")

    if not TEMPLATE_CSV.exists():
        print(f"ERROR: {TEMPLATE_CSV} no existe.")
        sys.exit(1)
    if not MASTER_CSV.exists():
        print(f"ERROR: {MASTER_CSV} no existe.")
        sys.exit(1)

    with open(MASTER_CSV, encoding="utf-8", newline="") as f:
        master_rows = list(csv.DictReader(f))

    # TAREA 3 — Validar antes de escribir
    print("  Validando consistencia...")
    issues = validate(LABELS, master_rows)
    if issues:
        print("  ERRORES encontrados — abortando:")
        for iss in issues:
            print(f"    {iss}")
        sys.exit(1)
    print("  Validacion OK (30 IDs, sin solapamientos, conteos correctos)")

    # TAREA 2 — Aplicar
    print("  Aplicando etiquetas...")
    rows, updated = apply_labels(LABELS)
    print(f"  Actualizadas: {updated} filas")

    # Conteos finales en template
    counts: dict[str, int] = {}
    for row in rows:
        rid = row["review_id"]
        if rid in LABELS:
            counts[row["human_label"]] = counts.get(row["human_label"], 0) + 1

    # TAREA 4 — Reporte
    status = write_report(LABELS, updated, issues)

    # TAREA 6 — Verificación final
    print()
    for p in [TEMPLATE_CSV, REPORT_MD]:
        ok = p.exists()
        print(f"  {'OK' if ok else 'MISSING':7} {p.name}")

    print()
    print("=== RESUMEN PACK 002 ===")
    print(f"  GOOD   : {counts.get('GOOD', 0)}")
    print(f"  BAD    : {counts.get('BAD', 0)}")
    print(f"  INVALID: {counts.get('INVALID', 0)}")
    print(f"  REVIEW : {counts.get('REVIEW', 0)}")
    print(f"  TOTAL  : {updated}")
    print(f"  STATUS : {status}")


if __name__ == "__main__":
    main()
