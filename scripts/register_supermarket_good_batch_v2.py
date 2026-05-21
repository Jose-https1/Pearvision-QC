"""
register_supermarket_good_batch_v2.py

Registra formalmente el segundo lote de supermercado como validacion humana GOOD.
Diagnostica errores por fondo y guarda hard examples para V3.

NO entrena ningun modelo.
NO modifica el dataset V2.
NO modifica best_model.pt, analyze_quality.py ni quality_rules.yaml.
"""

import csv
import shutil
import sys
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR     = PROJECT_ROOT / "outputs" / "supermarket_good_batch_v2_eval"
INPUT_DIR    = PROJECT_ROOT / "data"    / "unseen_quality_eval_input" / "supermarket_good_batch_v2"
HARD_DIR     = PROJECT_ROOT / "data"    / "supermarket_good_hard_examples_v2"
REPORTS_DIR  = PROJECT_ROOT / "reports"

PRED_CSV = EVAL_DIR / "predictions.csv"

# Falsos BAD confirmados por revision humana
FALSE_BAD_FILES = {
    "1000060770.jpg",
    "1000060771.jpg",
    "1000060773.jpg",
    "1000060774.jpg",
    "1000060775.jpg",
    "1000060779.jpg",
    "1000060781.jpg",
}

# Grupos de fondo según el usuario
BACKGROUND_GROUPS = {
    "white_light": {
        "1000060759.jpg", "1000060760.jpg", "1000060761.jpg", "1000060762.jpg",
        "1000060766.jpg", "1000060767.jpg", "1000060768.jpg", "1000060769.jpg",
    },
    "blue": {
        "1000060770.jpg", "1000060771.jpg", "1000060772.jpg",
        "1000060773.jpg", "1000060774.jpg", "1000060775.jpg",
    },
    "black_textured": {
        "1000060776.jpg", "1000060777.jpg", "1000060779.jpg", "1000060780.jpg",
        "1000060781.jpg", "1000060782.jpg", "1000060783.jpg", "1000060784.jpg",
    },
}

def get_background(filename):
    for group, files in BACKGROUND_GROUPS.items():
        if filename in files:
            return group
    return "unknown"


# ── TAREA 1: leer predicciones ─────────────────────────────────────────────────
def read_predictions():
    rows = []
    with PRED_CSV.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
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
    fields = ["image_id", "filename", "source_path", "pred_label",
              "pred_confidence", "human_label", "human_notes"]
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            if r["filename"] in FALSE_BAD_FILES:
                note = ("False BAD confirmado. Pera sana de supermercado. "
                        "El error parece causado por fondo/luz/sombra, no por defecto real.")
            else:
                note = "GOOD confirmado por revisión humana. Pera sana de supermercado."
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


# ── TAREA 3: human_error_review.csv ───────────────────────────────────────────
def write_error_review(rows):
    out = EVAL_DIR / "human_error_review.csv"
    fields = ["filename", "pred_label", "pred_confidence", "human_label",
              "is_model_error", "error_type", "background_group",
              "recommended_action", "notes"]
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            is_error = r["pred_label"] == "bad"
            bg       = get_background(r["filename"])
            unknown  = bg == "unknown"
            if unknown:
                print(f"  AVISO: {r['filename']} sin grupo de fondo asignado → unknown")
            if is_error:
                error_type = "FALSE_BAD_BACKGROUND_LIGHTING"
                action     = "ADD_AS_GOOD_HARD_EXAMPLE_FOR_V3"
                note       = (f"Falso BAD con conf={r['pred_confidence']:.4f}. "
                              f"Fondo {bg}. Pera sana mal clasificada por cambio de dominio visual.")
            else:
                error_type = ""
                action     = "ACCEPT"
                note       = "Prediccion correcta."
            w.writerow({
                "filename":           r["filename"],
                "pred_label":         r["pred_label"],
                "pred_confidence":    r["pred_confidence"],
                "human_label":        "GOOD",
                "is_model_error":     is_error,
                "error_type":         error_type,
                "background_group":   bg,
                "recommended_action": action,
                "notes":              note,
            })
    print(f"  [OK] {out.name}")
    return out


# ── TAREA 4: background_error_analysis.csv ────────────────────────────────────
def write_background_analysis(rows):
    out = EVAL_DIR / "background_error_analysis.csv"
    group_names = ["white_light", "blue", "black_textured", "unknown"]
    stats = {g: {"total": 0, "pred_good": 0, "pred_bad": 0,
                 "false_bad": 0, "conf_good": [], "conf_bad": []}
             for g in group_names}

    for r in rows:
        bg = get_background(r["filename"])
        if bg not in stats:
            stats[bg] = {"total": 0, "pred_good": 0, "pred_bad": 0,
                         "false_bad": 0, "conf_good": [], "conf_bad": []}
        s = stats[bg]
        s["total"] += 1
        if r["pred_label"] == "good":
            s["pred_good"] += 1
            s["conf_good"].append(r["pred_confidence"])
        else:
            s["pred_bad"] += 1
            s["conf_bad"].append(r["pred_confidence"])
            if r["filename"] in FALSE_BAD_FILES:
                s["false_bad"] += 1

    fields = ["background_group", "total_images", "pred_good", "pred_bad",
              "false_bad_count", "false_bad_rate",
              "mean_confidence_good", "mean_confidence_bad"]
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for g in group_names:
            s = stats[g]
            if s["total"] == 0:
                continue
            fb_rate  = round(s["false_bad"] / s["total"], 4)
            mean_g   = round(sum(s["conf_good"]) / len(s["conf_good"]), 4) if s["conf_good"] else "N/A"
            mean_b   = round(sum(s["conf_bad"])  / len(s["conf_bad"]),  4) if s["conf_bad"]  else "N/A"
            w.writerow({
                "background_group":    g,
                "total_images":        s["total"],
                "pred_good":           s["pred_good"],
                "pred_bad":            s["pred_bad"],
                "false_bad_count":     s["false_bad"],
                "false_bad_rate":      fb_rate,
                "mean_confidence_good": mean_g,
                "mean_confidence_bad":  mean_b,
            })
    print(f"  [OK] {out.name}")
    return stats


# ── TAREA 5: hard examples V2 ─────────────────────────────────────────────────
def create_hard_examples(rows):
    images_dir = HARD_DIR / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for r in rows:
        src = Path(r["source_path"])
        dst = images_dir / src.name
        if src.exists():
            shutil.copy2(src, dst)
            copied += 1
        else:
            print(f"  AVISO: no encontrado {src}")

    labels_path = HARD_DIR / "labels.csv"
    fields = ["filename", "source_path", "human_label", "origin",
              "background_group", "was_false_bad_v2", "notes"]
    with labels_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            is_fb = r["filename"] in FALSE_BAD_FILES
            bg    = get_background(r["filename"])
            if is_fb:
                note = (f"Falso BAD de V2 (conf={r['pred_confidence']:.4f}, fondo {bg}). "
                        "Hard example clave para enseñar dominio de fondo a V3.")
            else:
                note = f"GOOD confirmado. Fondo {bg}. Pera sana de supermercado."
            w.writerow({
                "filename":        r["filename"],
                "source_path":     r["source_path"],
                "human_label":     "GOOD",
                "origin":          "supermarket_spain_unseen_batch_v2",
                "background_group": bg,
                "was_false_bad_v2": is_fb,
                "notes":           note,
            })

    print(f"  [OK] {copied} imagenes copiadas → {images_dir}")
    print(f"  [OK] {labels_path.name}")
    return labels_path


# ── TAREA 6: capture_conditions_recommendation.txt ────────────────────────────
def write_capture_recommendation():
    out = EVAL_DIR / "capture_conditions_recommendation.txt"
    content = """\
=== Recomendaciones de condiciones de captura para PearVision QC ===

Basado en la evaluación del lote supermarket_good_batch_v2 (22 imágenes reales),
se ha observado que el clasificador V2 es sensible al fondo y la iluminación.

CONDICIONES RECOMENDADAS (fondo estable para V2 actual):
---------------------------------------------------------
  - Fondo blanco, gris claro o beige claro mate.
  - Iluminación difusa, uniforme, sin sombras duras.
  - Pera completa visible, centrada en el encuadre.
  - Sin manos ni objetos extraños en la imagen.
  - Sin fondos azules (tela, carpeta, superficie azul).
  - Sin fondos negros brillantes ni negros texturizados.
  - Sin maderas con veta fuerte ni piedras texturizadas
    durante la validación actual del sistema.

CONDICIONES QUE GENERAN FALSOS BAD EN V2:
------------------------------------------
  - Fondo azul (tasa de falso BAD observada: ~83%):
    5 de 6 imágenes en fondo azul fueron clasificadas incorrectamente como BAD.

  - Fondo negro/texturizado (tasa de falso BAD observada: ~25%):
    2 de 8 imágenes en fondo negro fueron clasificadas incorrectamente como BAD.

INTERPRETACIÓN:
---------------
  El problema NO es la calidad real de las peras.
  Las peras eran sanas y aptas para supermercado en todos los casos.
  El clasificador V2 fue entrenado principalmente con imágenes de fondo blanco/neutro
  (Fruits-360 dataset) y no generaliza bien a fondos de color intenso o texturas fuertes.

  Esto es un problema conocido de cambio de dominio visual (domain shift).

RECOMENDACIÓN OPERATIVA:
-------------------------
  Hasta que se entrene o valide una V3 robusta a fondos variados, o hasta que se aplique
  una máscara de segmentación de pera antes del clasificador:

    → Usar exclusivamente fondo blanco, gris claro o beige mate para captura de imágenes.
    → Tratar fondos azul y negro como "fuera de condición operativa" para V2.
    → Las predicciones BAD sobre imágenes en fondo azul/negro deben pasar a REVIEW
      manual, no a rechazo automático.

PRÓXIMO PASO SUGERIDO:
-----------------------
  Acumular más lotes con fondo blanco y fondos problemáticos para decidir si entrenar V3
  con los hard examples GOOD acumulados en:
    data/supermarket_good_hard_examples_v1/  (20 imágenes)
    data/supermarket_good_hard_examples_v2/  (22 imágenes)
"""
    out.write_text(content, encoding="utf-8")
    print(f"  [OK] {out.name}")
    return out


# ── TAREA 7: reporte final ─────────────────────────────────────────────────────
def write_report(rows, bg_stats):
    n_good  = sum(1 for r in rows if r["pred_label"] == "good")
    n_bad   = sum(1 for r in rows if r["pred_label"] == "bad")

    pred_table = "\n".join(
        f"| {r['image_id']} | {r['filename']} | {r['pred_label'].upper()} "
        f"| {r['pred_confidence']:.4f} | {get_background(r['filename'])} "
        f"| {'TRUE' if r['filename'] in FALSE_BAD_FILES else ''} |"
        for r in rows
    )

    bg_table = ""
    for g in ["white_light", "blue", "black_textured"]:
        s = bg_stats.get(g, {})
        if not s or s["total"] == 0:
            continue
        rate = f"{s['false_bad'] / s['total'] * 100:.1f}%"
        mg   = f"{sum(s['conf_good'])/len(s['conf_good']):.4f}" if s["conf_good"] else "N/A"
        mb   = f"{sum(s['conf_bad'])/len(s['conf_bad']):.4f}"   if s["conf_bad"]  else "N/A"
        bg_table += (f"| {g} | {s['total']} | {s['pred_good']} | {s['pred_bad']} "
                     f"| {s['false_bad']} | {rate} | {mg} | {mb} |\n")

    false_bad_list = "\n".join(
        f"| `{r['filename']}` | {r['pred_confidence']:.4f} | "
        f"{get_background(r['filename'])} | "
        f"{'Alta — problema de dominio' if r['pred_confidence'] >= 0.70 else 'Baja — zona de incertidumbre'} |"
        for r in rows if r["filename"] in FALSE_BAD_FILES
    )

    content = f"""# Supermarket Good Batch V2 — Review Report

**Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Lote:** `supermarket_spain_unseen_batch_v2`
**Modelo evaluado:** `outputs/fruits360_quality_cls_v2/best_model.pt` (V2, congelado)

---

## 1. Descripción del lote

- **22 fotos reales** de peras compradas en supermercado en España.
- Peras **comercialmente sanas**, aptas para venta.
- Características naturales presentes: manchas marrones, russeting, partes verdes,
  coloración irregular.
- **Variaciones de fondo en este lote:**
  - Fondo blanco/claro (8 imágenes)
  - Fondo azul / carpeta azul (6 imágenes)
  - Fondo negro/texturizado (8 imágenes)
- Verdad humana para todas: **GOOD**.

---

## 2. Resultado bruto del modelo V2

| Predicción | Cantidad | Confianza media |
|---|---|---|
| GOOD | {n_good} | {sum(r['pred_confidence'] for r in rows if r['pred_label'] == 'good') / n_good:.4f} |
| BAD  | {n_bad}  | {sum(r['pred_confidence'] for r in rows if r['pred_label'] == 'bad')  / n_bad:.4f}  |

- **Tasa de error bruta en este lote: {n_bad}/22 = {n_bad*100//22}%**
- Comparación con lote V1 (fondo más uniforme): 1/20 = 5%

---

## 3. Falsos BAD confirmados (7)

| filename | pred_confidence | background_group | gravedad del error |
|---|---|---|---|
{false_bad_list}

---

## 4. Diagnóstico por grupo de fondo

| background_group | total | pred_good | pred_bad | false_bad | false_bad_rate | conf_media_good | conf_media_bad |
|---|---|---|---|---|---|---|---|
{bg_table}

**Conclusión del diagnóstico:**

- **Fondo blanco/claro:** 0 falsos BAD. El modelo V2 es estable en este dominio.
- **Fondo azul:** 5/6 imágenes mal clasificadas (83%). El fondo azul interfiere
  con las características aprendidas por V2 (entrenado principalmente con Fruits-360,
  que usa fondo blanco uniforme).
- **Fondo negro/texturizado:** 2/8 imágenes mal clasificadas (25%). Mejor que azul
  pero aún problemático.
- **El problema es cambio de dominio visual (domain shift), no defectos reales.**

---

## 5. Predicciones completas

| id | filename | pred | conf | fondo | false_bad |
|---|---|---|---|---|---|
{pred_table}

---

## 6. Decisiones tomadas

1. **Registrar las 22 imágenes como hard examples GOOD** para futura V3:
   - `data/supermarket_good_hard_examples_v2/` (22 imágenes)
   - Junto con `data/supermarket_good_hard_examples_v1/` (20 imágenes) = 42 hard examples GOOD acumulados.
2. **No entrenar todavía.** 42 hard examples no son suficientes para reentrenar con garantías.
3. **Usar fondo blanco/gris/beige como condición operativa recomendada** para la demo actual.
4. **Aplicar regla operativa de umbral:** BAD con conf < 0.70 → REVIEW (no rechazo automático).
5. **Evaluar V3** cuando se acumulen al menos 100-150 hard examples GOOD con fondos variados,
   o cuando se implemente una máscara de segmentación de pera previa al clasificador.

---

## 7. Archivos generados

| Archivo | Descripción |
|---|---|
| `outputs/supermarket_good_batch_v2_eval/human_review_completed.csv` | Revisión humana con todas las 22 anotadas como GOOD |
| `outputs/supermarket_good_batch_v2_eval/human_error_review.csv` | Análisis de errores con grupo de fondo |
| `outputs/supermarket_good_batch_v2_eval/background_error_analysis.csv` | Estadísticas por grupo de fondo |
| `outputs/supermarket_good_batch_v2_eval/capture_conditions_recommendation.txt` | Guía de condiciones de captura |
| `data/supermarket_good_hard_examples_v2/labels.csv` | Etiquetas hard examples |
| `data/supermarket_good_hard_examples_v2/images/` | 22 imágenes copiadas |

---

## 8. Confirmaciones

- **NO** se entrenó ningún modelo.
- **NO** se modificó el dataset V2 (`data/quality_fruits360_human_v2/`).
- **NO** se modificó `best_model.pt`.
- **NO** se modificó `analyze_quality.py`.
- **NO** se modificó `quality_rules.yaml`.

---

## 9. Siguiente paso recomendado

Hacer una evaluación comparativa usando solo condiciones válidas de captura
(fondo blanco/gris/beige mate). Después decidir si entrenar V3 con los
hard examples GOOD acumulados en los dos lotes (42 imágenes en total).
"""
    out = REPORTS_DIR / "supermarket_good_batch_v2_review_report.md"
    out.write_text(content, encoding="utf-8")
    print(f"  [OK] {out.name}")
    return out


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=== register_supermarket_good_batch_v2 ===")

    rows   = read_predictions()
    n_good = sum(1 for r in rows if r["pred_label"] == "good")
    n_bad  = sum(1 for r in rows if r["pred_label"] == "bad")
    fb_present = [fn for fn in FALSE_BAD_FILES if any(r["filename"] == fn for r in rows)]

    print(f"  Predicciones leidas: {len(rows)}  (GOOD={n_good}, BAD={n_bad})")
    print(f"  Falsos BAD confirmados presentes: {len(fb_present)}/7")
    unknown = [r["filename"] for r in rows if get_background(r["filename"]) == "unknown"]
    if unknown:
        print(f"  AVISO: sin grupo de fondo asignado: {unknown}")
    print()

    print("  TAREA 2: human_review_completed.csv ...")
    write_human_review_completed(rows)

    print("  TAREA 3: human_error_review.csv ...")
    write_error_review(rows)

    print("  TAREA 4: background_error_analysis.csv ...")
    bg_stats = write_background_analysis(rows)

    print("  TAREA 5: hard examples V2 ...")
    create_hard_examples(rows)

    print("  TAREA 6: capture_conditions_recommendation.txt ...")
    write_capture_recommendation()

    print("  TAREA 7: reporte final ...")
    write_report(rows, bg_stats)

    print()
    print("=" * 58)
    print("SUPERMARKET GOOD BATCH V2 REGISTRADO")
    print()
    print("Total imágenes revisadas: 22")
    print("Verdad humana: todas GOOD")
    print()
    print("Resultado bruto V2:")
    print(f"- GOOD: {n_good}")
    print(f"- BAD: {n_bad}")
    print()
    print("Falsos BAD confirmados:")
    for fn in sorted(FALSE_BAD_FILES):
        r = next((x for x in rows if x["filename"] == fn), None)
        conf = f"conf={r['pred_confidence']:.4f}" if r else "N/A"
        bg   = get_background(fn)
        print(f"  - {fn}  ({conf}, fondo {bg})")
    print()
    print("Conclusión:")
    print("- V2 funciona razonablemente bien en fondo blanco/claro.")
    print("- V2 falla con más frecuencia en fondo azul y fondo negro/texturizado.")
    print("- El problema principal es cambio de fondo/luz, no defecto real de la fruta.")
    print("- El lote queda guardado como hard examples GOOD para futura V3.")
    print("- Para la demo actual, se recomienda fondo blanco/gris/beige mate e iluminación controlada.")
    print()
    print("Archivos creados:")
    print("- outputs/supermarket_good_batch_v2_eval/human_review_completed.csv")
    print("- outputs/supermarket_good_batch_v2_eval/human_error_review.csv")
    print("- outputs/supermarket_good_batch_v2_eval/background_error_analysis.csv")
    print("- outputs/supermarket_good_batch_v2_eval/capture_conditions_recommendation.txt")
    print("- data/supermarket_good_hard_examples_v2/labels.csv")
    print("- reports/supermarket_good_batch_v2_review_report.md")
    print()
    print("NO se entrenó ningún modelo.")
    print("NO se modificó V2.")
    print("NO se modificó analyze_quality.py.")
    print("NO se modificó quality_rules.yaml.")
    print()
    print("Siguiente paso recomendado:")
    print("hacer una evaluación comparativa usando solo condiciones válidas de captura:")
    print("fondo blanco/gris/beige mate. Después decidir si entrenar V3 con hard examples GOOD.")


if __name__ == "__main__":
    main()
