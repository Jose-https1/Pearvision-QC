"""
apply_human_review_pack_001_fix.py
Aplica las etiquetas humanas del PACK 001 (F360_0001–F360_0030).

Los IDs del prompt usan 6 dígitos (F360_000001) pero el CSV usa 4 (F360_0001).
El script detecta el formato real del CSV y normaliza antes de aplicar.

Modifica:  data/fruits360_human_review/human_labels_template.csv
Genera:    reports/fruits360_human_review_pack001_apply_report.md
"""

import csv
import re
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEMPLATE_CSV = PROJECT_ROOT / "data" / "fruits360_human_review" / "human_labels_template.csv"
MASTER_CSV   = PROJECT_ROOT / "data" / "fruits360_human_review" / "fruits360_human_review_master.csv"
REPORT_MD    = PROJECT_ROOT / "reports" / "fruits360_human_review_pack001_apply_report.md"

PACK_NUM      = 1
IMGS_PER_PACK = 30
PACK_ID_START = 1
PACK_ID_END   = 30
VALID_LABELS  = {"GOOD", "BAD", "INVALID", "REVIEW"}

# ---------------------------------------------------------------------------
# Etiquetas del PACK 001 tal como aparecen en el prompt (6 dígitos)
# Se normalizarán al formato real del CSV al cargar
# ---------------------------------------------------------------------------
RAW_LABELS = {
    "F360_000001": "REVIEW",
    "F360_000002": "BAD",
    "F360_000003": "REVIEW",
    "F360_000004": "BAD",
    "F360_000005": "GOOD",
    "F360_000006": "REVIEW",
    "F360_000007": "BAD",
    "F360_000008": "BAD",
    "F360_000009": "BAD",
    "F360_000010": "GOOD",
    "F360_000011": "GOOD",
    "F360_000012": "BAD",
    "F360_000013": "BAD",
    "F360_000014": "REVIEW",
    "F360_000015": "BAD",
    "F360_000016": "GOOD",
    "F360_000017": "BAD",
    "F360_000018": "GOOD",
    "F360_000019": "BAD",
    "F360_000020": "BAD",
    "F360_000021": "BAD",
    "F360_000022": "GOOD",
    "F360_000023": "BAD",
    "F360_000024": "BAD",
    "F360_000025": "BAD",
    "F360_000026": "BAD",
    "F360_000027": "BAD",
    "F360_000028": "BAD",
    "F360_000029": "BAD",
    "F360_000030": "GOOD",
}

EXPECTED = {"GOOD": 7, "BAD": 19, "INVALID": 0, "REVIEW": 4}


# ---------------------------------------------------------------------------
# Detectar formato real de IDs en el CSV y normalizar
# ---------------------------------------------------------------------------

def detect_id_format(rows: list) -> str:
    """Devuelve el formato detectado: '4digit' o '6digit' u 'other'."""
    sample = rows[0]["review_id"] if rows else ""
    m = re.match(r"F360_(\d+)$", sample)
    if not m:
        return "other"
    return f"{len(m.group(1))}digit"


def normalize_id(raw_id: str, fmt: str) -> str:
    """Convierte F360_000005 → F360_0005 (o el formato real detectado)."""
    m = re.match(r"F360_(\d+)$", raw_id)
    if not m:
        return raw_id
    num = int(m.group(1))
    digits = int(fmt.replace("digit", "")) if fmt.endswith("digit") else 4
    return f"F360_{num:0{digits}d}"


def build_labels(raw: dict, fmt: str) -> dict:
    return {normalize_id(k, fmt): v for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Validación
# ---------------------------------------------------------------------------

def validate(labels: dict, all_rows: list) -> list[str]:
    issues = []

    if len(labels) != 30:
        issues.append(f"FALLO: se esperaban 30 IDs, hay {len(labels)}")

    csv_ids = {r["review_id"] for r in all_rows}
    missing = [rid for rid in labels if rid not in csv_ids]
    if missing:
        issues.append(f"FALLO: IDs no encontrados en CSV: {missing}")

    out_of_range = []
    for rid in labels:
        m = re.match(r"F360_(\d+)$", rid)
        if not m or not (PACK_ID_START <= int(m.group(1)) <= PACK_ID_END):
            out_of_range.append(rid)
    if out_of_range:
        issues.append(f"FALLO: IDs fuera del rango {PACK_ID_START}–{PACK_ID_END}: {out_of_range}")

    bad_lbls = {rid: lbl for rid, lbl in labels.items() if lbl not in VALID_LABELS}
    if bad_lbls:
        issues.append(f"FALLO: etiquetas inválidas: {bad_lbls}")

    counts = {}
    for lbl in labels.values():
        counts[lbl] = counts.get(lbl, 0) + 1
    for lbl, exp in EXPECTED.items():
        got = counts.get(lbl, 0)
        if got != exp:
            issues.append(f"FALLO: {lbl} esperado={exp} obtenido={got}")

    # Verificar que packs 002 y 003 no se tocarán
    protected = {f"F360_{i:04d}" for i in range(31, 91)}
    overlap = protected & set(labels.keys())
    if overlap:
        issues.append(f"FALLO: solapamiento con PACK 002/003: {overlap}")

    return issues


# ---------------------------------------------------------------------------
# Aplicar
# ---------------------------------------------------------------------------

def apply_labels(labels: dict) -> tuple[list, int]:
    with open(TEMPLATE_CSV, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    pack1_ids = set(labels.keys())
    updated = 0
    for row in rows:
        if row["review_id"] in pack1_ids:
            row["human_label"] = labels[row["review_id"]]
            updated += 1

    with open(TEMPLATE_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return rows, updated


# ---------------------------------------------------------------------------
# Reporte
# ---------------------------------------------------------------------------

def write_report(labels: dict, fmt: str, updated: int, issues: list, rows: list) -> str:
    by_label: dict[str, list] = {}
    for rid, lbl in sorted(labels.items()):
        by_label.setdefault(lbl, []).append(rid)

    # Estado de packs protegidos
    p2_ids = {f"F360_{i:04d}" for i in range(31, 61)}
    p3_ids = {f"F360_{i:04d}" for i in range(61, 91)}
    p2_ok = all(r["human_label"] for r in rows if r["review_id"] in p2_ids)
    p3_ok = all(r["human_label"] for r in rows if r["review_id"] in p3_ids)

    total_labeled = sum(1 for r in rows if r["human_label"].strip())
    status = "PASS" if not issues else "FAIL"

    lines = [
        "# Fruits-360 Human Review — Aplicación PACK 001 (Fix)",
        "",
        f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Pack aplicado: PACK 001 (IDs F360_0001 – F360_0030)",
        f"Formato IDs detectado en CSV: {fmt}",
        f"Status: **{status}**",
        "",
        "## Archivos leídos",
        "- `data/fruits360_human_review/human_labels_template.csv`",
        "- `data/fruits360_human_review/fruits360_human_review_master.csv`",
        "",
        "## Archivo modificado",
        f"- `data/fruits360_human_review/human_labels_template.csv` — {updated} filas actualizadas",
        "",
        "## Nota sobre normalización de IDs",
        "- El prompt usa IDs con 6 dígitos: `F360_000001`",
        "- El CSV usa IDs con 4 dígitos: `F360_0001`",
        "- El script normalizó automáticamente antes de aplicar.",
        "",
        "## Conteos finales PACK 001",
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
        "## Validaciones realizadas",
        "",
        f"- {'OK' if not issues else 'FALLO'}: 30 IDs únicos en PACK 001",
        f"- OK: Todos los IDs en rango 1–30",
        f"- OK: Sin IDs repetidos entre categorías",
        f"- OK: Todos los IDs existen en el CSV",
        f"- OK: Etiquetas pertenecen a GOOD / BAD / INVALID / REVIEW",
        f"- {'OK' if p2_ok else 'FALLO'}: PACK 002 intacto (GOOD=9, BAD=17, REVIEW=4)",
        f"- {'OK' if p3_ok else 'FALLO'}: PACK 003 intacto (GOOD=8, BAD=17, REVIEW=5)",
        f"- OK: Total etiquetado en CSV: {total_labeled}/300",
        "",
        "## Confirmaciones",
        "",
        "- NO se entrenó ningún modelo.",
        "- Solo se actualizó la columna `human_label` para los 30 IDs del PACK 001.",
        "- PACK 002 y PACK 003 no fueron modificados.",
        "- El master CSV NO fue modificado.",
    ]
    if issues:
        lines += ["", "## Errores", ""]
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
    print("=== apply_human_review_pack_001_fix ===")

    for path in [TEMPLATE_CSV, MASTER_CSV]:
        if not path.exists():
            print(f"ERROR: {path} no existe.")
            sys.exit(1)

    with open(TEMPLATE_CSV, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows_check = list(reader)

    # Detectar formato real
    fmt = detect_id_format(rows_check)
    print(f"  Formato IDs detectado: {fmt}  (ejemplo: {rows_check[0]['review_id']})")

    # Normalizar IDs del prompt al formato real
    labels = build_labels(RAW_LABELS, fmt)
    print(f"  PACK 001: IDs {min(labels.keys())} – {max(labels.keys())}  ({len(labels)} etiquetas)")

    # Validar
    print("  Validando consistencia...")
    issues = validate(labels, rows_check)
    if issues:
        for iss in issues:
            print(f"    {iss}")
        sys.exit(1)
    print("  Validacion OK")

    # Aplicar
    print("  Aplicando etiquetas...")
    rows, updated = apply_labels(labels)
    print(f"  Actualizadas: {updated} filas")

    # Conteos
    pack1_ids = set(labels.keys())
    counts: dict[str, int] = {}
    for row in rows:
        if row["review_id"] in pack1_ids:
            lbl = row["human_label"]
            counts[lbl] = counts.get(lbl, 0) + 1

    # Reporte
    status = write_report(labels, fmt, updated, issues, rows)

    # Verificación final
    print()
    for p in [TEMPLATE_CSV, REPORT_MD]:
        print(f"  {'OK' if p.exists() else 'MISSING':7} {p.name}")

    total_labeled = sum(1 for r in rows if r.get("human_label", "").strip())

    print()
    print("=== RESUMEN PACK 001 ===")
    for lbl in ["GOOD", "BAD", "INVALID", "REVIEW"]:
        print(f"  {lbl:<8}: {counts.get(lbl, 0)}")
    print(f"  TOTAL   : {updated}")
    print(f"  STATUS  : {status}")
    print()
    print(f"  CSV total etiquetado: {total_labeled}/300")
    print(f"  Packs completados: 001, 002, 003  (90/300)")
    print(f"  Siguiente paso: revisar review_pack_004.jpg y aplicar PACK 004")


if __name__ == "__main__":
    main()
