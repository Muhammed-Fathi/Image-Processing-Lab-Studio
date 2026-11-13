"""Prepare the six approved source photos and the derived low-contrast input.

Usage (from repository root):
    python scripts/prepare_final_assets.py --download
    python scripts/prepare_final_assets.py --source-dir "D:\Path\To\Pictures"

The --source-dir path may contain the photographer's original Wikimedia
filenames, or simply: poppies.jpg, floral.jpg, sunset.jpg, compass.jpg,
rock.jpg, museum.jpg. Run from a Windows virtual environment with the
project's requirements installed. No data is collected or uploaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.parse
import urllib.request

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import image_operations as ops  # noqa: E402

TARGET = ROOT / "assets" / "samples"
FILES = {
    "poppies": ("Red_poppies_under_morning_sun_(Unsplash).jpg", "poppies_rgb_grayscale.jpg", 1280),
    "floral": ("Dark_Floral_Tones_(Unsplash).jpg", "dark_floral_tones.jpg", 960),
    "sunset": ("Tree_silhouette_sunset._(Unsplash).jpg", "sunset_solarization.jpg", 1280),
    "compass": ("Blue_color_cast_-_Compass_1.jpg", "blue_cast_compass.jpg", 960),
    "rock": ("Rock_1,_Elizabeth_Castle,_Jersey.jpg", "rock_original.jpg", 1280),
    "museum": ('Colorful_"Museum”_sign_on_landscape_(Unsplash).jpg', "museum_clean.jpg", 1280),
}
# Wikimedia title for the museum file contains a curly closing quote (U+201D)
# as shown on its Commons file page; the API handles it as a Unicode title.


def open_url(url: str) -> bytes:
    request = urllib.request.Request(url, headers={
        "User-Agent": "ImageProcessingLabStudio/1.0 (university learning project; MediaWiki media reuse)",
        "Accept": "application/json, image/jpeg, */*",
    })
    with urllib.request.urlopen(request, timeout=40) as res:
        return res.read()


def download_from_commons(file_title: str, width: int) -> bytes:
    params = urllib.parse.urlencode({
        "action": "query",
        "format": "json",
        "prop": "imageinfo",
        "iiprop": "url",
        "iiurlwidth": width,
        "titles": "File:" + file_title,
    })
    data = json.loads(open_url("https://commons.wikimedia.org/w/api.php?" + params))
    pages = list(data["query"]["pages"].values())
    if not pages or "missing" in pages[0]:
        raise RuntimeError(f"Commons file page not found: {file_title}")
    info = pages[0]["imageinfo"][0]
    return open_url(info.get("thumburl") or info["url"])


def locate_local(source_dir: Path, key: str) -> bytes:
    files = sorted(
        p for p in source_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    # Match a short name or a descriptive Wikimedia source filename.
    matches = [p for p in files if key.lower() in p.stem.lower()]
    if not matches:
        raise FileNotFoundError(
            f"No photo containing '{key}' was found in '{source_dir}'. "
            f"Rename your file to '{key}.jpg' if necessary."
        )
    if len(matches) > 1:
        raise ValueError(
            f"More than one candidate for '{key}': "
            + ", ".join(p.name for p in matches)
            + ". Keep just one candidate in your source folder."
        )
    return matches[0].read_bytes()


def decode_and_resize(data: bytes, width: int) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The source file is not a readable photograph.")
    if image.shape[1] > width:
        h = max(1, round(image.shape[0] * width / image.shape[1]))
        image = cv2.resize(image, (width, h), interpolation=cv2.INTER_AREA)
    return image


def save_jpeg(path: Path, bgr: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
    if not ok:
        raise RuntimeError(f"Cannot encode {path}")
    path.write_bytes(encoded.tobytes())


def prepare(source_dir: Path | None = None) -> dict:
    TARGET.mkdir(parents=True, exist_ok=True)
    inputs: dict[str, np.ndarray] = {}
    manifest: dict = {"sources": {}, "derived": {}}
    for key, (title, out_name, width) in FILES.items():
        if source_dir:
            raw = locate_local(source_dir, key)
        else:
            raw = download_from_commons(title, width)
        img = decode_and_resize(raw, width)
        path = TARGET / out_name
        save_jpeg(path, img)
        inputs[key] = img
        manifest["sources"][key] = {
            "filename": out_name,
            "commons_page": "https://commons.wikimedia.org/wiki/File:" + title,
            "size": [int(img.shape[1]), int(img.shape[0])],
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        print(f"Prepared {path.relative_to(ROOT)} ({img.shape[1]}x{img.shape[0]})")

    low = ops.make_low_contrast(inputs["rock"], band=(105, 165))
    low_path = TARGET / "rock_low_contrast.png"
    if not cv2.imwrite(str(low_path), low):
        raise RuntimeError("Failed to write rock_low_contrast.png")
    manifest["derived"]["task6_low_contrast"] = {
        "filename": low_path.name,
        "parent": "rock_original.jpg",
        "operation": "src.image_operations.make_low_contrast(image, band=(105, 165))",
        "sha256": hashlib.sha256(low_path.read_bytes()).hexdigest(),
        "observed_channel_range": [int(low.min()), int(low.max())],
        "note": "Synthetic contrast reduction applied to a real photograph; not an unedited photo.",
    }

    target = ROOT / "assets" / "final_assets_manifest.json"
    target.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nCreated {target.relative_to(ROOT)}")
    print("Photos ready. Now run the tests and inspect the nine pages before committing.")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--download", action="store_true", help="Download all photos from Wikimedia Commons.")
    modes.add_argument("--source-dir", type=Path, help="Folder containing the six original photos.")
    args = parser.parse_args()
    if args.source_dir and not args.source_dir.is_dir():
        parser.error(f"Folder not found: {args.source_dir}")
    try:
        prepare(args.source_dir)
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        parser.exit(1, f"Asset preparation failed: {error}\n"
                    "If downloads are blocked, save the six photos locally and "
                    'run --source-dir "path_to_photos".\n')


if __name__ == "__main__":
    main()
