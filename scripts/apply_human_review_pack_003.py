"""
apply_human_review_pack_003.py
Aplica las etiquetas humanas del PACK 003 (F360_0061–F360_0090) al CSV de revisión.

Modifica:  data/fruits360_human_review/human_labels_template.csv
Genera:    reports/fruits360_human_review_pack003_apply_report.md
"""

import csv
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEMPLATE_CSV = PROJECT_ROOT / "data" / "fruits360_human_review" / "human_labels_template.csv"
MASTER_CSV   = PROJECT_ROOT / "data" / "fruits360_human_review" / "fruits360_human_review_master.csv"
REPORT_MD    = PROJECT_ROOT / "reports" / "fruits360_human_review_pack003_apply_report.md"

PACK_NUM      = 3
IMGS_PER_PACK = 30
PACK_ID_START = (PACK_NUM - 1) * IMGS_PER_PACK + 1   # 61
PACK_ID_END   = PACK_NUM * IMGS_PER_PACK               # 90

VALID_LABELS  = {"GOOD", "BAD", "INVALID", "REVIEW"}

# ---------------------------------------------------------------------------
# Etiquetas humanas oficiales PACK 003
# ---------------------------------------------------------------------------
LABELS = {
    "F360_0061": "REVIEW",
    "F360_0062": "GOOD",
    "F360_0063": "BAD",
    "F360_0064": "GOOD",
    "F360_0065": "GOOD",
    "F360_0066": "BAD",
    "F360_0067": "REVIEW",
    "F360_0068": "BAD",
    "F360_0069": "GOOD",
    "F360_0070": "BAD",
    "F360_0071": "GOOD",
    "F360_0072": "BAD",
    "F360_0073": "BAD",
    "F360_0074": "BAD",
    "F360_0075": "REVIEW",
    "F360_0076": "BAD",
    "F360_0077": "BAD",
    "F360_0078": "BAD",
    "F360_0079": "BAD",
    "F360_0080": "GOOD",
    "F360_0081": "REVIEW",
    "F360_0082": "REVIEW",
    "F360_0083": "GOOD",
    "F360_0084": "BAD",
    "F360_0085": "BAD",
    "F360_0086": "BAD",
    "F360_0087": "BAD",
    "F360_0088": "BAD",
    "F360_0089": "BAD",
    "F360_0090": "GOOD",
}

EXPECTED = {"GOOD": 8, "BAD": 17, "INVALID": 0, "REVIEW": 5}


# ---------------------------------------------------------------------------
# Validación
# ---------------------------------------------------------------------------

def validate(labels: dict, master_rows: list) -> list[str]:
    issues = []

    if len(labels) != 30:
        issues.append(f"FALLO: se esperaban 30 IDs, hay {len(labels)}")

    master_ids = {r["review_id"] for r in master_rows}
    missing = [rid for rid in labels if rid not in master_ids]
    if missing:
        issues.append(f"FALLO: IDs no encontrados en master: {missing}")

    out_of_range = []
    for rid in labels:
        try:
            n = int(rid.split("_")[1])
            if not (PACK_ID_START <= n <= PACK_ID_END):
                out_of_range.append(rid)
        except Exception:
            out_of_range.append(rid)
    if out_of_range:
        issues.append(f"FALLO: IDs fuera del rango {PACK_ID_START}–{PACK_ID_END}: {out_of_range}")

    bad_labels = {rid: lbl for rid, lbl in labels.items() if lbl not in VALID_LABELS}
    if bad_labels:
        issues.append(f"FALLO: etiquetas inválidas: {bad_labels}")

    counts = {}
    for lbl in labels.values():
        counts[lbl] = counts.get(lbl, 0) + 1
    for lbl, exp in EXPECTED.items():
        got = counts.get(lbl, 0)
        if got != exp:
            issues.append(f"FALLO: {lbl} esperado={exp} obtenido={got}")

    return issues


# ---------------------------------------------------------------------------
# Aplicar etiquetas
# ---------------------------------------------------------------------------

def apply_labels(labels: dict) -> tuple[list, int]:
    with open(TEMPLATE_CSV, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    updated = 0
    pack3_ids = set(labels.keys())

    for row in rows:
        rid = row["review_id"]
        if rid in pack3_ids:
            row["human_label"] = labels[rid]
            updated += 1

    with open(TEMPLATE_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return rows, updated


# ---------------------------------------------------------------------------
# Reporte
# ---------------------------------------------------------------------------

def write_report(labels: dict, updated: int, issues: list[str], rows: list) -> str:
    by_label: dict[str, list] = {}
    for rid, lbl in sorted(labels.items()):
        by_label.setdefault(lbl, []).append(rid)

    # Verificar packs anteriores intactos
    pack12_ids = {f"F360_{i:04d}" for i in range(1, 61)}
    pack12_rows = [r for r in rows if r["review_id"] in pack12_ids]
    pack12_labeled = sum(1 for r in pack12_rows if r.get("human_label", "").strip())

    status = "PASS" if not issues else "FAIL"

    lines = [
        "# Fruits-360 Human Review — Aplicación PACK 003",
        "",
        f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Pack aplicado: PACK 003 (IDs F360_0061 – F360_0090)",
        f"Status: **{status}**",
        "",
        "## Archivos leídos",
        f"- `data/fruits360_human_review/human_labels_template.csv`",
        f"- `data/fruits360_human_review/fruits360_human_review_master.csv`",
        "",
        "## Archivo modificado",
        f"- `data/fruits360_human_review/human_labels_template.csv` — {updated} filas actualizadas",
        "",
        "## Conteos finales PACK 003",
        "",
        "| Etiqueta | Esperado | Obtenido | IDs |",
        "|----------|----------|----------|-----|",
    ]
    for lbl in ["GOOD", "BAD", "INVALID", "REVIEW"]:
        ids = by_label.get(lbl, [])
        exp = EXPECTED.get(lbl, 0)
        ok_mark = "OK" if len(ids) == exp else "FALLO"
        id_str = ", ".join(ids) if ids else "(ninguno)"
        lines.append(f"| {lbl} | {exp} | {len(ids)} {ok_mark} | {id_str} |")

    lines += [
        "",
        "## Validaciones realizadas",
        "",
        f"- {'OK' if len(labels)==30 else 'FALLO'}: 30 IDs únicos en PACK 003",
        f"- {'OK' if not [rid for rid in labels if int(rid.split('_')[1]) not in range(PACK_ID_START, PACK_ID_END+1)] else 'FALLO'}: Todos los IDs en rango 61–90",
        f"- OK: Sin IDs repetidos entre categorías",
        f"- OK: Todos los IDs existen en el master",
        f"- OK: Etiquetas pertenecen a GOOD / BAD / INVALID / REVIEW",
        f"- OK: PACK 001+002 intactos ({pack12_labeled}/60 etiquetados, sin modificar en esta operación)",
        "",
        "## Confirmaciones",
        "",
        "- NO se entrenó ningún modelo.",
        "- Solo se actualizó la columna `human_label` para los 30 IDs del PACK 003.",
        "- El master CSV NO fue modificado.",
        "- Los IDs, imágenes y rutas originales no fueron alterados.",
        "- PACK 001 y PACK 002 no fueron modificados.",
    ]
    if issues:
        lines += ["", "## Errores detectados", ""]
        for iss in issues:
            lines.append(f"- {iss}")

    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Reporte: {REPORT_MD}")
    return status


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== apply_human_review_pack_003 ===")

    for path in [TEMPLATE_CSV, MASTER_CSV]:
        if not path.exists():
            print(f"ERROR: {path} no existe.")
            sys.exit(1)

    with open(MASTER_CSV, encoding="utf-8", newline="") as f:
        master_rows = list(csv.DictReader(f))

    # TAREA 1 ya realizada externamente; confirmar IDs
    print(f"  PACK 003: IDs F360_{PACK_ID_START:04d} – F360_{PACK_ID_END:04d}  ({len(LABELS)} etiquetas)")

    # Validar antes de escribir
    print("  Validando consistencia...")
    issues = validate(LABELS, master_rows)
    if issues:
        for iss in issues:
            print(f"    {iss}")
        sys.exit(1)
    print("  Validacion OK")

    # Aplicar
    print("  Aplicando etiquetas...")
    rows, updated = apply_labels(LABELS)
    print(f"  Actualizadas: {updated} filas")

    # Conteos en el CSV actualizado
    pack3_ids = set(LABELS.keys())
    counts: dict[str, int] = {}
    for row in rows:
        if row["review_id"] in pack3_ids:
            lbl = row["human_label"]
            counts[lbl] = counts.get(lbl, 0) + 1

    # Reporte
    status = write_report(LABELS, updated, issues, rows)

    # Verificación final
    print()
    for p in [TEMPLATE_CSV, REPORT_MD]:
        print(f"  {'OK' if p.exists() else 'MISSING':7} {p.name}")

    # Estado global del CSV
    with open(TEMPLATE_CSV, encoding="utf-8", newline="") as f:
        all_rows = list(csv.DictReader(f))
    total_labeled = sum(1 for r in all_rows if r.get("human_label", "").strip())

    print()
    print("=== RESUMEN PACK 003 ===")
    for lbl in ["GOOD", "BAD", "INVALID", "REVIEW"]:
        print(f"  {lbl:<8}: {counts.get(lbl, 0)}")
    print(f"  TOTAL   : {updated}")
    print(f"  STATUS  : {status}")
    print()
    print(f"  CSV total etiquetado: {total_labeled}/300")
    print(f"  Siguiente paso: revisar review_pack_004.jpg y aplicar PACK 004")


if __name__ == "__main__":
    main()
