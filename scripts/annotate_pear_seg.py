"""
annotate_pear_seg.py - Anotador de segmentacion de pera para PearVision QC.

Genera labels YOLO segmentation a partir de clicks sobre el contorno de la pera.
Las coordenadas se normalizan respecto a la imagen original y se guardan en
data/pear_seg/labels/ junto con una copia de la imagen en data/pear_seg/images/.

Uso:
    python scripts/annotate_pear_seg.py --source data/samples_quality_controlled_test
    python scripts/annotate_pear_seg.py --source data/samples

Controles:
    Click izquierdo : anadir punto al poligono
    u               : deshacer ultimo punto
    r               : reiniciar todos los puntos
    s               : guardar label y pasar a la siguiente imagen
    n               : saltar imagen sin guardar
    q               : salir
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEST_IMAGES = PROJECT_ROOT / "data" / "pear_seg" / "images"
DEST_LABELS = PROJECT_ROOT / "data" / "pear_seg" / "labels"

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_DISPLAY_PX = 1024


# ---------------------------------------------------------------------------
# Unicode-safe I/O (Windows paths con caracteres especiales)
# ---------------------------------------------------------------------------

def imread_unicode(path):
    data = np.fromfile(str(Path(path)), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar: {path}")
    return img


def imwrite_unicode(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".jpg"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        raise IOError(f"No se pudo codificar: {path}")
    buf.tofile(str(path))


# ---------------------------------------------------------------------------
# Colectar imagenes
# ---------------------------------------------------------------------------

def collect_images(source: Path) -> list:
    if source.is_file():
        return [source]
    imgs = []
    for ext in VALID_EXT:
        imgs.extend(source.glob(f"*{ext}"))
        imgs.extend(source.glob(f"*{ext.upper()}"))
    return sorted(set(imgs))


# ---------------------------------------------------------------------------
# Visualizacion
# ---------------------------------------------------------------------------

FONT = cv2.FONT_HERSHEY_SIMPLEX
_INSTRUCCIONES = [
    "Click izq: punto  |  u: deshacer  |  r: reiniciar",
    "s: guardar+sig    |  n: saltar    |  q: salir",
]


def _safe(text):
    return text.encode("ascii", errors="replace").decode("ascii")


def _draw_frame(base, points, img_name, progress, status_msg):
    vis = base.copy()
    h, w = vis.shape[:2]

    # Relleno y contorno del poligono si hay >= 3 puntos
    if len(points) >= 3:
        arr = np.array(points, dtype=np.int32)
        ov = vis.copy()
        cv2.fillPoly(ov, [arr], (0, 180, 0))
        cv2.addWeighted(ov, 0.22, vis, 0.78, 0, vis)
        cv2.polylines(vis, [arr], isClosed=True, color=(0, 220, 0), thickness=2)

    # Lineas entre puntos consecutivos
    for i in range(1, len(points)):
        cv2.line(vis, points[i - 1], points[i], (0, 200, 0), 1)

    # Puntos marcados
    for pt in points:
        cv2.circle(vis, pt, 5, (0, 255, 255), -1)
        cv2.circle(vis, pt, 5, (0, 0, 0), 1)

    # Barra superior: progreso + nombre de imagen
    bar = f"{progress}  {_safe(img_name)}"
    cv2.putText(vis, bar, (8, 22), FONT, 0.52, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(vis, bar, (8, 22), FONT, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    cnt = f"Puntos: {len(points)}"
    cv2.putText(vis, cnt, (8, 44), FONT, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(vis, cnt, (8, 44), FONT, 0.48, (180, 220, 255), 1, cv2.LINE_AA)

    # Mensaje de estado (encima de instrucciones)
    if status_msg:
        sy = h - 14 - len(_INSTRUCCIONES) * 20
        cv2.putText(vis, _safe(status_msg), (8, sy), FONT, 0.46, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, _safe(status_msg), (8, sy), FONT, 0.46, (50, 230, 255), 1, cv2.LINE_AA)

    # Instrucciones en la parte inferior
    for i, line in enumerate(_INSTRUCCIONES):
        y = h - 8 - (len(_INSTRUCCIONES) - 1 - i) * 20
        cv2.putText(vis, line, (8, y), FONT, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, line, (8, y), FONT, 0.42, (190, 190, 190), 1, cv2.LINE_AA)

    return vis


# ---------------------------------------------------------------------------
# Guardar label YOLO segmentation
# ---------------------------------------------------------------------------

def _save_label(img_path: Path, display_img, points: list, scale: float):
    """Guarda la imagen de display y el label YOLO segmentation.

    Los puntos de click son del espacio display (puede estar redimensionado).
    Se convierten al espacio original divididendo por scale, luego se
    normalizan por las dimensiones originales.
    """
    dh, dw = display_img.shape[:2]
    orig_w = int(round(dw / scale))
    orig_h = int(round(dh / scale))

    coords = []
    for (x, y) in points:
        coords.append(f"{(x / scale) / orig_w:.6f}")
        coords.append(f"{(y / scale) / orig_h:.6f}")

    label_line = "0 " + " ".join(coords) + "\n"

    dest_img = DEST_IMAGES / img_path.name
    imwrite_unicode(dest_img, display_img)

    dest_lbl = DEST_LABELS / (img_path.stem + ".txt")
    dest_lbl.write_text(label_line, encoding="utf-8")

    print(f"  [OK]  {img_path.name}  ->  {dest_lbl.name}  ({len(points)} puntos)")


# ---------------------------------------------------------------------------
# Bucle principal de anotacion
# ---------------------------------------------------------------------------

def run(source: Path):
    DEST_IMAGES.mkdir(parents=True, exist_ok=True)
    DEST_LABELS.mkdir(parents=True, exist_ok=True)

    images = collect_images(source)
    if not images:
        print(f"ERROR: no se encontraron imagenes en {source}")
        sys.exit(1)

    total = len(images)
    print(f"Imagenes encontradas : {total}")
    print(f"Destino              : {DEST_IMAGES.parent}")
    print(f"Las ya anotadas se saltan automaticamente.")
    print()

    WIN = "PearSeg Annotator"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 900, 700)

    for idx, img_path in enumerate(images):
        # Saltar si ya existe el label
        dest_lbl = DEST_LABELS / (img_path.stem + ".txt")
        if dest_lbl.exists():
            print(f"  [SKIP] {img_path.name}  (ya anotada, label existe)")
            continue

        progress = f"[{idx + 1}/{total}]"
        print(f"{progress} {img_path.name}")

        try:
            orig = imread_unicode(img_path)
        except Exception as e:
            print(f"  ERROR cargando imagen: {e}")
            continue

        oh, ow = orig.shape[:2]
        scale = 1.0
        if max(oh, ow) > MAX_DISPLAY_PX:
            scale = MAX_DISPLAY_PX / max(oh, ow)
            display = cv2.resize(orig, (int(ow * scale), int(oh * scale)),
                                 interpolation=cv2.INTER_AREA)
        else:
            display = orig.copy()

        points = []
        status_msg = ""
        action = None

        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))

        cv2.setMouseCallback(WIN, on_mouse)

        while True:
            frame = _draw_frame(display, points, img_path.name, progress, status_msg)
            cv2.imshow(WIN, frame)
            key = cv2.waitKey(30) & 0xFF

            if key == ord("u"):
                if points:
                    points.pop()
                    status_msg = "Punto eliminado"
                else:
                    status_msg = "Sin puntos que deshacer"
            elif key == ord("r"):
                points.clear()
                status_msg = "Reiniciado"
            elif key == ord("s"):
                if len(points) < 3:
                    status_msg = f"Minimo 3 puntos (ahora: {len(points)})"
                else:
                    _save_label(img_path, display, points, scale)
                    action = "save"
                    break
            elif key == ord("n"):
                print(f"  [SKIP] {img_path.name}  (saltada sin guardar)")
                action = "skip"
                break
            elif key == ord("q"):
                action = "quit"
                break

        if action == "quit":
            print("Saliendo del anotador.")
            break

    cv2.destroyAllWindows()
    annotated = len(list(DEST_LABELS.glob("*.txt")))
    print()
    print(f"Anotador finalizado  ->  {annotated} label(s) en {DEST_LABELS}")


# ---------------------------------------------------------------------------
# Entrada
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PearSeg Annotator — anotador de segmentacion YOLO para peras"
    )
    parser.add_argument(
        "--source", type=Path, required=True,
        help="Carpeta (o imagen) con las imagenes a anotar"
    )
    args = parser.parse_args()

    source = (PROJECT_ROOT / args.source).resolve()
    if not source.exists():
        print(f"ERROR: no existe: {source}")
        sys.exit(1)

    run(source)


if __name__ == "__main__":
    main()
