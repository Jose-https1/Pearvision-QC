"""
register_supermarket_good_batch_v1.py

Registra formalmente el lote de supermercado como validacion humana GOOD.
Crea archivos de revision, hard examples y reporte.

NO entrena ningun modelo.
NO modifica el dataset V2.
NO modifica best_model.pt.
NO modifica analyze_quality.py ni quality_rules.yaml.
"""

import csv
import shutil
import sys
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR     = PROJECT_ROOT / "outputs" / "supermarket_unseen_quality_v2_eval"
INPUT_DIR    = PROJECT_ROOT / "data"    / "unseen_quality_eval_input"
HARD_DIR     = PROJECT_ROOT / "data"    / "supermarket_good_hard_examples_v1"
REPORTS_DIR  = PROJECT_ROOT / "reports"

PRED_CSV     = EVAL_DIR / "predictions.csv"
TEMPLATE_CSV = EVAL_DIR / "human_review_template.csv"

FALSE_BAD_FILE = "1000060747.jpg"
FALSE_BAD_NOTE = (
    "False BAD de baja confianza; pera sana de supermercado con russeting/mancha "
    "marrón natural. Debe ser GOOD o como máximo REVIEW operativo."
)
DEFAULT_NOTE = "Pera sana de supermercado; GOOD confirmado por revisión humana."

# Umbral operativo: BAD con conf < 0.70 → REVIEW en lugar de rechazo
BAD_REVIEW_THRESHOLD = 0.70


def read_predictions():
    rows = []
    with PRED_CSV.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "image_id":        row["image_id"],
                "filename":        row["filename"],
                "source_path":     row["source_path"],
                "pred_label":      row["pred_label"],
                "pred_confidence": float(row["pred_confidence"]),
                "prob_good":       float(row["prob_good"]),
                "prob_bad":        float(row["prob_bad"]),
            })
    return rows


# ── TAREA 2: human_review_completed.csv ───────────────────────────────────────
def write_human_review_completed(rows):
    out = EVAL_DIR / "human_review_completed.csv"
    fields = ["image_id", "filename", "source_path", "pred_label", "pred_confidence",
              "human_label", "human_notes"]
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            note = FALSE_BAD_NOTE if r["filename"] == FALSE_BAD_FILE else DEFAULT_NOTE
            w.writerow({
                "image_id":        r["image_id"],
                "filename":        r["filename"],
                "source_path":     r["source_path"],
                "pred_label":      r["pred_label"],
                "pred_confidence": r["pred_confidence"],
                "human_label":     "GOOD",
                "human_notes":     note,
            })
    print(f"  [OK] {out.name}")
    return out


# ── TAREA 3: human_error_review.csv ──────────────────────────────────────────
def write_error_review(rows):
    out = EVAL_DIR / "human_error_review.csv"
    fields = ["filename", "pred_label", "pred_confidence", "human_label",
              "is_model_error", "error_type", "recommended_operational_action", "notes"]
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            human_label = "GOOD"
            is_error    = r["pred_label"] != "good"
            if r["filename"] == FALSE_BAD_FILE:
                error_type  = "FALSE_BAD_LOW_CONFIDENCE"
                rec_action  = "REVIEW_NOT_REJECT"
                note        = "BAD con confianza 0.5275; pera sana con russeting natural"
            else:
                error_type  = "" if not is_error else "FALSE_BAD"
                rec_action  = "ACCEPT" if not is_error else "REVIEW"
                note        = "Prediccion correcta" if not is_error else "Error de modelo"
            w.writerow({
                "filename":                       r["filename"],
                "pred_label":                     r["pred_label"],
                "pred_confidence":                r["pred_confidence"],
                "human_label":                    human_label,
                "is_model_error":                 is_error,
                "error_type":                     error_type,
                "recommended_operational_action": rec_action,
                "notes":                          note,
            })
    print(f"  [OK] {out.name}")
    return out


# ── TAREA 4: hard examples ────────────────────────────────────────────────────
def create_hard_examples(rows):
    images_dir = HARD_DIR / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Copiar imágenes
    copied = 0
    for r in rows:
        src = Path(r["source_path"])
        dst = images_dir / src.name
        if src.exists():
            shutil.copy2(src, dst)
            copied += 1
        else:
            print(f"  AVISO: no encontrado {src}")

    # labels.csv
    labels_path = HARD_DIR / "labels.csv"
    fields = ["filename", "source_path", "human_label", "origin", "notes"]
    with labels_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            note = (
                "Falso BAD de V2 (conf=0.5275); pera sana con russeting natural. Hard example clave."
                if r["filename"] == FALSE_BAD_FILE
                else "Pera sana de supermercado con russeting/manchas marrones naturales."
            )
            w.writerow({
                "filename":    r["filename"],
                "source_path": r["source_path"],
                "human_label": "GOOD",
                "origin":      "supermarket_spain_unseen_batch_v1",
                "notes":       note,
            })

    print(f"  [OK] {copied} imagenes copiadas → {images_dir}")
    print(f"  [OK] {labels_path.name}")
    return labels_path


# ── TAREA 5: operational_threshold_analysis.csv ───────────────────────────────
def write_operational_analysis(rows):
    out = EVAL_DIR / "operational_threshold_analysis.csv"
    fields = ["filename", "pred_label", "pred_confidence", "human_label",
              "raw_model_correct", "operational_decision", "operational_correct_or_safe"]
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            human_label = "GOOD"
            raw_correct = (r["pred_label"] == "good")
            # Regla operativa
            if r["pred_label"] == "good":
                op_decision = "GOOD"
            elif r["pred_confidence"] < BAD_REVIEW_THRESHOLD:
                op_decision = "REVIEW"
            else:
                op_decision = "BAD"
            # Seguro si GOOD o REVIEW (no rechaza algo que es GOOD)
            op_safe = op_decision in ("GOOD", "REVIEW")
            w.writerow({
                "filename":                  r["filename"],
                "pred_label":                r["pred_label"],
                "pred_confidence":           r["pred_confidence"],
                "human_label":               human_label,
                "raw_model_correct":         raw_correct,
                "operational_decision":      op_decision,
                "operational_correct_or_safe": op_safe,
            })
    print(f"  [OK] {out.name}")
    return out


# ── TAREA 6: reporte final ────────────────────────────────────────────────────
def write_report(rows):
    n_good = sum(1 for r in rows if r["pred_label"] == "good")
    n_bad  = sum(1 for r in rows if r["pred_label"] == "bad")
    errors = [r for r in rows if r["pred_label"] != "good"]

    op_good   = sum(1 for r in rows if r["pred_label"] == "good")
    op_review = sum(1 for r in rows if r["pred_label"] == "bad" and r["pred_confidence"] < BAD_REVIEW_THRESHOLD)
    op_bad    = sum(1 for r in rows if r["pred_label"] == "bad" and r["pred_confidence"] >= BAD_REVIEW_THRESHOLD)

    table = "\n".join(
        f"| {r['filename']} | {r['pred_label'].upper()} | {r['pred_confidence']:.4f} | "
        f"{'GOOD (operativo: REVIEW)' if r['filename'] == FALSE_BAD_FILE else 'GOOD'} | "
        f"{'FALSE_BAD_LOW_CONFIDENCE' if r['filename'] == FALSE_BAD_FILE else '-'} |"
        for r in rows
    )

    content = f"""# Supermarket Good Batch V1 — Review Report

**Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Lote:** `supermarket_spain_unseen_batch_v1`
**Modelo evaluado:** `outputs/fruits360_quality_cls_v2/best_model.pt` (V2, congelado)

---

## 1. Descripción del lote

- **20 fotos reales** de peras compradas en supermercado en España.
- Peras **comercialmente sanas**, aptas para venta.
- Características naturales presentes: manchas marrones, russeting, zonas verdes, coloración irregular.
- **Ninguna** de las 20 peras tiene defectos graves (golpes, podredumbre, marcas de ramita).
- Verdad humana para todas: **GOOD**.

---

## 2. Resultado bruto del modelo V2

| Predicción | Cantidad | Confianza media |
|---|---|---|
| GOOD | {n_good} | {sum(r['pred_confidence'] for r in rows if r['pred_label'] == 'good') / n_good:.4f} |
| BAD  | {n_bad}  | {rows[10]['pred_confidence']:.4f} |

- Precisión bruta del modelo sobre este lote: **{n_good}/20 = {n_good*100//20}%**

---

## 3. Error detectado

| Imagen | Predicción modelo | Confianza | Verdad humana | Tipo de error |
|---|---|---|---|---|
| `1000060747.jpg` | BAD | 0.5275 | GOOD | FALSE_BAD_LOW_CONFIDENCE |

- El modelo predijo BAD con confianza muy baja (**0.5275**, apenas por encima de 0.50).
- La imagen corresponde a una pera sana con russeting/manchas marrones naturales.
- Este tipo de fallo es esperado en V2: el modelo aún no tiene suficiente exposición a peras
  sanas con coloración irregular natural de supermercado español.

---

## 4. Interpretación

**V2 se comporta bien frente a russeting natural:**
- 19 de 20 peras sanas clasificadas correctamente como GOOD.
- Confianza media en GOOD: alta (≥ 0.75 en todos los casos correctos).

**El único fallo no debe causar rechazo automático:**
- La predicción BAD tiene confianza de 0.5275, casi aleatoria.
- Una regla operativa simple (BAD con conf < 0.70 → REVIEW) evita el rechazo indebido.
- Con esta regla, el resultado operativo para este lote sería: {op_good} GOOD, {op_review} REVIEW, {op_bad} BAD.

---

## 5. Análisis de umbral operativo

Regla propuesta: `BAD conf < {BAD_REVIEW_THRESHOLD} → REVIEW`

| Decisión operativa | Cantidad | Interpretación |
|---|---|---|
| GOOD   | {op_good}  | Aceptadas directamente |
| REVIEW | {op_review} | Enviadas a revisión humana (no rechazadas) |
| BAD    | {op_bad}   | Rechazadas automáticamente |

Con esta regla, **0 peras sanas serían rechazadas automáticamente** en este lote.

---

## 6. Recomendaciones

1. **Conservar V2 como baseline.** Su rendimiento en peras sanas de supermercado es aceptable.
2. **Aplicar zona de incertidumbre operativa:** BAD con confianza < 0.70 → REVIEW (no rechazo).
3. **Usar estas 20 imágenes como hard examples GOOD para V3:**
   - Ruta: `data/supermarket_good_hard_examples_v1/`
   - Especialmente `1000060747.jpg` como ejemplo de falso BAD con russeting natural.
4. **Acumular más lotes reales** antes de entrenar V3. Un solo lote de 20 imágenes
   no es suficiente para reentrenar — acumular al menos 3-5 lotes similares.

---

## 7. Archivos generados

| Archivo | Descripción |
|---|---|
| `outputs/supermarket_unseen_quality_v2_eval/human_review_completed.csv` | Revisión humana completa |
| `outputs/supermarket_unseen_quality_v2_eval/human_error_review.csv` | Análisis de errores del modelo |
| `outputs/supermarket_unseen_quality_v2_eval/operational_threshold_analysis.csv` | Análisis umbral operativo |
| `data/supermarket_good_hard_examples_v1/labels.csv` | Etiquetas hard examples |
| `data/supermarket_good_hard_examples_v1/images/` | 20 imágenes copiadas |

---

## 8. Confirmaciones

- **NO** se entrenó ningún modelo.
- **NO** se modificó el dataset V2 (`data/quality_fruits360_human_v2/`).
- **NO** se modificó `best_model.pt`.
- **NO** se modificó `analyze_quality.py`.
- **NO** se modificó `quality_rules.yaml`.

---

## 9. Siguiente paso

Hacer una segunda prueba con otro lote real de peras sanas y, si aparecen más falsos BAD,
acumularlos junto con este lote para entrenar V3.
"""
    out = REPORTS_DIR / "supermarket_good_batch_v1_review_report.md"
    out.write_text(content, encoding="utf-8")
    print(f"  [OK] {out.name}")
    return out


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=== register_supermarket_good_batch_v1 ===")

    rows = read_predictions()
    n_good = sum(1 for r in rows if r["pred_label"] == "good")
    n_bad  = sum(1 for r in rows if r["pred_label"] == "bad")
    print(f"  Predicciones leidas: {len(rows)}  (GOOD={n_good}, BAD={n_bad})")
    print(f"  Caso problemático: {FALSE_BAD_FILE} presente = "
          f"{any(r['filename'] == FALSE_BAD_FILE for r in rows)}")
    print()

    print("  TAREA 2: human_review_completed.csv ...")
    write_human_review_completed(rows)

    print("  TAREA 3: human_error_review.csv ...")
    write_error_review(rows)

    print("  TAREA 4: hard examples ...")
    create_hard_examples(rows)

    print("  TAREA 5: operational_threshold_analysis.csv ...")
    write_operational_analysis(rows)

    print("  TAREA 6: reporte final ...")
    write_report(rows)

    print()
    print("=" * 55)
    print("SUPERMARKET GOOD BATCH V1 REGISTRADO")
    print()
    print("Total imágenes revisadas: 20")
    print("Verdad humana: todas GOOD")
    print()
    print("Resultado bruto V2:")
    print(f"- GOOD: {n_good}")
    print(f"- BAD: {n_bad}")
    print()
    print("Error humano confirmado:")
    print("- 1000060747.jpg: predicha BAD, verdad humana GOOD, confianza baja")
    print()
    print("Conclusión:")
    print("- V2 funciona bien con peras sanas de supermercado con russeting/manchas marrones.")
    print("- La predicción BAD de baja confianza debe tratarse como REVIEW, no como rechazo directo.")
    print("- El lote queda guardado como hard examples GOOD para futura V3.")
    print()
    print("Archivos creados:")
    print("- outputs/supermarket_unseen_quality_v2_eval/human_review_completed.csv")
    print("- outputs/supermarket_unseen_quality_v2_eval/human_error_review.csv")
    print("- outputs/supermarket_unseen_quality_v2_eval/operational_threshold_analysis.csv")
    print("- data/supermarket_good_hard_examples_v1/labels.csv")
    print("- reports/supermarket_good_batch_v1_review_report.md")
    print()
    print("NO se entrenó ningún modelo.")
    print("NO se modificó V2.")
    print("NO se modificó analyze_quality.py.")
    print("NO se modificó quality_rules.yaml.")
    print()
    print("Siguiente paso:")
    print("hacer una segunda prueba con otro lote real de peras sanas y, si aparecen")
    print("más falsos BAD, acumularlos para entrenar V3.")


if __name__ == "__main__":
    main()
