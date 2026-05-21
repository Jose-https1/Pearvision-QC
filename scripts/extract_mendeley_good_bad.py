from pathlib import Path
from zipfile import ZipFile
import shutil
import re

zip_path = Path("data_external/mendeley_good_bad_pear/mendeley_good_bad_pear.zip")
out_root = Path("data_external/mendeley_good_bad_pear/raw_clean")

if not zip_path.exists():
    raise FileNotFoundError(f"No existe el ZIP: {zip_path}")

valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
counts = {"good": 0, "bad": 0, "skipped": 0}

def safe_filename(name: str) -> str:
    name = name.strip().rstrip(" .")
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
    return name or "image.jpg"

def unique_path(folder: Path, filename: str) -> Path:
    dest = folder / filename
    if not dest.exists():
        return dest

    stem = dest.stem
    suffix = dest.suffix
    i = 1
    while True:
        candidate = folder / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1

out_root.mkdir(parents=True, exist_ok=True)

with ZipFile(zip_path, "r") as z:
    for info in z.infolist():
        if info.is_dir():
            continue

        internal = info.filename.replace("\\", "/")
        parts = [p.strip().rstrip(" .") for p in internal.split("/") if p.strip()]
        if not parts:
            counts["skipped"] += 1
            continue

        ext = Path(parts[-1]).suffix.lower()
        if ext not in valid_ext:
            counts["skipped"] += 1
            continue

        label = None
        for p in parts:
            low = p.lower().strip().rstrip(" .")
            if low == "good":
                label = "good"
            elif low == "bad":
                label = "bad"

        if label is None:
            counts["skipped"] += 1
            continue

        dest_dir = out_root / label
        dest_dir.mkdir(parents=True, exist_ok=True)

        filename = safe_filename(parts[-1])
        dest = unique_path(dest_dir, filename)

        with z.open(info) as src, open(dest, "wb") as dst:
            shutil.copyfileobj(src, dst)

        counts[label] += 1

print("Extraccion limpia completada")
print(f"GOOD: {counts['good']}")
print(f"BAD : {counts['bad']}")
print(f"SKIP: {counts['skipped']}")
print(f"Salida: {out_root.resolve()}")
