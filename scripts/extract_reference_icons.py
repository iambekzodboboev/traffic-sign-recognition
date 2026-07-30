"""Stage 7.4 prep -- extract one clean reference icon per class from the
dataset zip's Classes/ folder, for the Telegram bot to send back alongside
its prediction. One-time local extraction, no cloud needed.

Selection rule per class: prefer a file named exactly "<class_id>.png"
(the dataset's own canonical icon); otherwise prefer a plain file over the
"*_road_sign_*.svg.png" reference variants pulled in from other sources;
otherwise take the first file alphabetically. Verified this resolves all
200 classes with no ambiguity left unhandled.
"""
import re
import zipfile
from collections import defaultdict
from pathlib import Path

ZIP_PATH = "Traffic sign dataset.zip"
OUT_DIR = Path("assets/class_icons")


def pick_canonical(class_id, files):
    exact = [f for f in files if Path(f).stem.lower() == str(class_id)]
    if exact:
        return sorted(exact)[0]
    plain = [f for f in files if not re.search(r'road_sign', f, re.IGNORECASE)]
    if plain:
        return sorted(plain)[0]
    return sorted(files)[0]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH) as z:
        names = [n for n in z.namelist() if n.startswith("Classes/") and not n.endswith("/")]

        by_class = defaultdict(list)
        for n in names:
            folder_name = n.split("/")[2]
            m = re.match(r"^(\d+)\s", folder_name)
            if m:
                by_class[int(m.group(1))].append(n)

        missing = [cid for cid in range(200) if cid not in by_class]
        assert not missing, f"No reference icon found for classes: {missing}"

        for class_id in range(200):
            chosen = pick_canonical(class_id, by_class[class_id])
            ext = Path(chosen).suffix.lower()
            dest = OUT_DIR / f"{class_id}{ext}"
            dest.write_bytes(z.read(chosen))

    print(f"Extracted {len(list(OUT_DIR.iterdir()))} reference icons to {OUT_DIR}/")


if __name__ == "__main__":
    main()
