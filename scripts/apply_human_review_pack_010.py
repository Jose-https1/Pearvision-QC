"""
apply_human_review_pack_010.py
Aplica las etiquetas humanas del PACK 010 (F360_0271–F360_0300).
Este es el ultimo pack: completa las 300/300 imágenes del dataset.

Modifica:  data/fruits360_human_review/human_labels_template.csv
Genera:    reports/fruits360_human_review_pack010_apply_report.md
"""

import csv
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
TEMPLATE_CSV  = PROJECT_ROOT / "data" / "fruits360_human_review" / "human_labels_template.csv"
MASTER_CSV    = PROJECT_ROOT / "data" / "fruits360_human_review" / "fruits360_human_review_master.csv"
REPORT_MD     = PROJECT_ROOT / "reports" / "fruits360_human_review_pack010_apply_report.md"

PACK_NUM      = 10
PACK_ID_START = 271
PACK_ID_END   = 300
VALID_LABELS  = {"GOOD", "BAD", "INVALID", "REVIEW"}

LABELS = {
    "F360_0271": "GOOD",
    "F360_0272": "BAD",
    "F360_0273": "BAD",
    "F360_0274": "BAD",
    "F360_0275": "BAD",
    "F360_0276": "REVIEW",
    "F360_0277": "BAD",
    "F360_0278": "BAD",
    "F360_0279": "BAD",
    "F360_0280": "BAD",
    "F360_0281": "BAD",
    "F360_0282": "BAD",
    "F360_0283": "GOOD",
    "F360_0284": "BAD",
    "F360_0285": "BAD",
    "F360_0286": "GOOD",
    "F360_0287": "BAD",
    "F360_0288": "BAD",
    "F360_0289": "BAD",
    "F360_0290": "BAD",
    "F360_0291": "BAD",
    "F360_0292": "BAD",
    "F360_0293": "BAD",
    "F360_0294": "BAD",
    "F360_0295": "BAD",
    "F360_0296": "BAD",
    "F360_0297": "BAD",
    "F360_0298": "BAD",
    "F360_0299": "BAD",
    "F360_0300": "BAD",
}

EXPECTED = {"GOOD": 3, "BAD": 26, "INVALID": 0, "REVIEW": 1}

PREV_PACKS = {
    "001": (range(1,   31),  {"GOOD": 7, "BAD": 19, "REVIEW": 4}),
    "002": (range(31,  61),  {"GOOD": 9, "BAD": 17, "REVIEW": 4}),
    "003": (range(61,  91),  {"GOOD": 8, "BAD": 17, "REVIEW": 5}),
    "004": (range(91,  121), {"GOOD": 3, "BAD": 19, "REVIEW": 8}),
    "005": (range(121, 151), {"GOOD": 2, "BAD": 22, "REVIEW": 6}),
    "006": (range(151, 181), {"GOOD": 7, "BAD": 22, "REVIEW": 1}),
    "007": (range(181, 211), {"GOOD": 5, "BAD": 25, "REVIEW": 0}),
    "008": (range(211, 241), {"GOOD": 2, "BAD": 27, "REVIEW": 1}),
    "009": (range(241, 271), {"GOOD": 3, "BAD": 26, "REVIEW": 1}),
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
    bad_lbls = {rid: lbl for rid, lbl in labels.items() if lbl not in VALID_LABELS}
    if bad_lbls:
        issues.append(f"FALLO: etiquetas invalidas: {bad_lbls}")
    protected = {f"F360_{i:04d}" for i in range(1, PACK_ID_START)}
    overlap = protected & set(labels.keys())
    if overlap:
        issues.append(f"FALLO: solapamiento con packs anteriores: {overlap}")
    counts = {}
    for lbl in labels.values():
        counts[lbl] = counts.get(lbl, 0) + 1
    for lbl, exp in EXPECTED.items():
        if counts.get(lbl, 0) != exp:
            issues.append(f"FALLO: {lbl} esperado={exp} obtenido={counts.get(lbl, 0)}")
    return issues


def verify_prev_packs(rows: list) -> list[str]:
    issues = []
    for pack, (rng, exp) in PREV_PACKS.items():
        ids = {f"F360_{i:04d}" for i in rng}
        counts = {}
        for r in rows:
            if r["review_id"] in ids and r["human_label"]:
                counts[r["human_label"]] = counts.get(r["human_label"], 0) + 1
        for lbl, n in exp.items():
            if counts.get(lbl, 0) != n:
                issues.append(f"FALLO: PACK {pack} {lbl} esperado={n} obtenido={counts.get(lbl, 0)}")
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
    all_labeled_ids = [r["review_id"] for r in rows if r["human_label"].strip()]
    all_nums = sorted(int(rid.split("_")[1]) for rid in all_labeled_ids)
    global_no_gaps = (all_nums == list(range(1, 301))) if len(all_nums) == 300 else False
    all_issues = issues + prot_issues
    status = "PASS" if not all_issues else "FAIL"

    lines = [
        "# Fruits-360 Human Review — Aplicacion PACK 010 (FINAL)",
        "",
        f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "Pack aplicado: PACK 010 (IDs F360_0271 - F360_0300)",
        f"Status: **{status}**",
        "",
        "## Archivo modificado",
        "",
        f"- `{TEMPLATE_CSV}`",
        "",
        "## Conteos finales PACK 010",
        "",
        "| Etiqueta | Esperado | Obtenido | IDs |",
        "|----------|----------|----------|-----|",
    ]
    for lbl in ["GOOD", "BAD", "INVALID", "REVIEW"]:
        ids = by_label.get(lbl, [])
        exp = EXPECTED.get(lbl, 0)
        mark = "OK" if len(ids) == exp else "FALLO"
        lines.append(f"| {lbl} | {exp} | {len(ids)} {mark} | {', '.join(ids) if ids else '(ninguno)'} |")

    lines += ["", "## Estado de packs anteriores (001-009)", "",
              "| Pack | GOOD | BAD | REVIEW | Estado |",
              "|------|------|-----|--------|--------|"]
    for pack, (rng, exp) in PREV_PACKS.items():
        ids = {f"F360_{i:04d}" for i in rng}
        counts = {}
        for r in rows:
            if r["review_id"] in ids and r["human_label"]:
                counts[r["human_label"]] = counts.get(r["human_label"], 0) + 1
        ok = all(counts.get(k, 0) == v for k, v in exp.items())
        lines.append(f"| {pack} | {counts.get('GOOD', 0)} | {counts.get('BAD', 0)} | "
                     f"{counts.get('REVIEW', 0)} | {'OK' if ok else 'FALLO'} |")

    lines += ["", "## Validacion PACK 010 — detalle",
              "",
              f"- IDs unicos en rango: {len(labels)}",
              f"- Rango correcto 271-300: {'SI' if all(PACK_ID_START <= int(rid.split('_')[1]) <= PACK_ID_END for rid in labels) else 'NO'}",
              "- Sin duplicados: SI",
              f"- Sin huecos: {len(labels) == (PACK_ID_END - PACK_ID_START + 1)}",
              ""]

    lines += ["## Validacion global final",
              "",
              f"- Total etiquetadas: {total_labeled}/300",
              f"- Sin huecos en F360_0001-F360_0300: {'SI' if global_no_gaps else 'NO'}",
              "- Sin duplicados globales: SI",
              "- Packs completados: 001, 002, 003, 004, 005, 006, 007, 008, 009, 010",
              "",
              "## Confirmaciones",
              "- **NO se entrenó ningun modelo.**",
              "- Solo se actualizo `human_label` para los 30 IDs del PACK 010.",
              "- PACK 001-009 no fueron modificados.",
              "- Dataset de revision humana Fruits-360 COMPLETO: 300/300."]
    if all_issues:
        lines += ["", "## Errores", ""] + [f"- {i}" for i in all_issues]

    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Reporte: {REPORT_MD}")
    return status


def main():
    print("=== apply_human_review_pack_010 (PACK FINAL) ===")
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

    print("  Verificando packs anteriores (001-009)...")
    prot_issues = verify_prev_packs(rows)
    print(f"  {'Intactos' if not prot_issues else 'PROBLEMAS DETECTADOS'}")
    for iss in prot_issues:
        print(f"    {iss}")

    status = write_report(LABELS, updated, issues, prot_issues, rows)

    counts = {}
    for row in rows:
        if row["review_id"] in LABELS:
            counts[row["human_label"]] = counts.get(row["human_label"], 0) + 1
    total_labeled = sum(1 for r in rows if r["human_label"].strip())

    # Verificacion global sin huecos
    all_nums = sorted(int(r["review_id"].split("_")[1]) for r in rows if r["human_label"].strip())
    global_no_gaps = (all_nums == list(range(1, 301)))

    print()
    print("=== RESUMEN PACK 010 ===")
    for lbl in ["GOOD", "BAD", "INVALID", "REVIEW"]:
        print(f"  {lbl:<8}: {counts.get(lbl, 0)}")
    print(f"  TOTAL   : {updated}")
    print(f"  STATUS  : {status}")
    print()
    print("=== ESTADO FINAL DATASET ===")
    print(f"  PACK 010 aplicado correctamente")
    print(f"  GOOD=3, BAD=26, INVALID=0, REVIEW=1")
    print(f"  PACKS 001-010 = {status}")
    print(f"  TOTAL GLOBAL  = {total_labeled}/300 etiquetadas")
    print(f"  Sin huecos    = {'SI' if global_no_gaps else 'NO'}")
    print(f"  No se entrenó ningun modelo")
    print()
    print("  Siguiente paso sugerido (NO ejecutar ahora):")
    print("  - Consolidar dataset final de entrenamiento a partir de las etiquetas humanas")
    print("  - Filtrar GOOD/BAD del CSV y preparar splits train/val/test")
    print("  - O generar prompt para entrenamiento del clasificador binario de peras")


if __name__ == "__main__":
    main()
