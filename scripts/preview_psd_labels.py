import cv2
import numpy as np
from pathlib import Path

def imread_unicode(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def imwrite_unicode(path, img):
    ext = Path(path).suffix
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(str(path))

root = Path("data_external/PSD/datasets/Date_demo")
classes_path = Path("data_external/PSD/datasets/classes.txt")
out = Path("outputs/psd_label_preview")
out.mkdir(parents=True, exist_ok=True)

classes = classes_path.read_text(encoding="utf-8").splitlines()

colors = [
    (0, 255, 0),      # pear
    (0, 0, 255),      # bruise
    (255, 0, 0),      # stab
    (0, 255, 255),    # twig
    (255, 0, 255),    # tcm
    (0, 128, 255),    # rot
]

count = 0

for split in ["train", "val"]:
    img_dir = root / "images" / split
    lab_dir = root / "labels" / split

    for img_path in img_dir.glob("*.jpg"):
        label_path = lab_dir / (img_path.stem + ".txt")

        img = imread_unicode(img_path)
        if img is None or not label_path.exists():
            continue

        h, w = img.shape[:2]

        lines = label_path.read_text(encoding="utf-8").strip().splitlines()

        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                continue

            cls = int(float(parts[0]))
            xc, yc, bw, bh = map(float, parts[1:])

            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)

            color = colors[cls % len(colors)]
            name = classes[cls] if cls < len(classes) else str(cls)

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                img,
                name,
                (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

        save_path = out / f"{split}_{img_path.name}"
        imwrite_unicode(save_path, img)
        count += 1

print("Previews guardadas en:", out)
print("Imagenes generadas:", count)
