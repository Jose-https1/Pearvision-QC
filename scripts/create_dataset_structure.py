from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FOLDERS = [
    "data/raw/own/session_001/good_pear",
    "data/raw/own/session_001/mechanical_damage",
    "data/raw/own/session_001/rot",
    "data/raw/own/session_001/twig_mark",
    "data/raw/external",
    "data/annotations/yolo/images",
    "data/annotations/yolo/labels",
    "data/annotations/metadata",
    "data/processed",
    "data/train/images",
    "data/train/labels",
    "data/val/images",
    "data/val/labels",
    "data/test/images",
    "data/test/labels",
]


def main():
    print(f"Raíz del proyecto: {PROJECT_ROOT}\n")
    created = 0
    confirmed = 0
    for folder in FOLDERS:
        path = PROJECT_ROOT / folder
        if path.exists():
            print(f"  [OK]     {folder}")
            confirmed += 1
        else:
            path.mkdir(parents=True, exist_ok=True)
            print(f"  [CREADA] {folder}")
            created += 1

    print(f"\nResumen: {created} carpeta(s) creada(s), {confirmed} ya existían.")


if __name__ == "__main__":
    main()
