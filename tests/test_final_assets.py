"""Tests for reproducible preparation of the approved final lab photos."""
from __future__ import annotations

import json

import cv2
import numpy as np

from scripts import prepare_final_assets as prep


def test_final_asset_registry_has_six_distinct_images():
    assert len(prep.FILES) == 6
    names = [spec[1] for spec in prep.FILES.values()]
    assert len(set(names)) == 6
    assert all(name.endswith(".jpg") for name in names)


def test_prepare_final_assets_from_local_photos(tmp_path, monkeypatch):
    """Offline, deterministic preparation; no Wikimedia requests required."""
    source = tmp_path / "user-originals"
    source.mkdir()
    for idx, key in enumerate(prep.FILES):
        # Include a gradient so histogram compression has a meaningful range.
        arr = np.zeros((32, 48, 3), dtype=np.uint8)
        arr[:, :, 0] = np.arange(48, dtype=np.uint8)[None, :] * 4
        arr[:, :, 1] = idx * 20 + 30
        arr[:, :, 2] = 230 - idx * 13
        ok, encoded = cv2.imencode(".jpg", arr)
        assert ok
        (source / f"{key}.jpg").write_bytes(encoded.tobytes())

    monkeypatch.setattr(prep, "ROOT", tmp_path)
    monkeypatch.setattr(prep, "TARGET", tmp_path / "assets" / "samples")
    output = prep.prepare(source_dir=source)
    final_dir = tmp_path / "assets" / "samples"
    assert len(output["sources"]) == 6
    for item in output["sources"].values():
        assert (final_dir / item["filename"]).is_file()
    derived = final_dir / "rock_low_contrast.png"
    assert derived.exists()
    loaded = cv2.imread(str(derived), cv2.IMREAD_COLOR)
    assert loaded is not None
    assert int(loaded.min()) >= 105
    assert int(loaded.max()) <= 165
    manifest = json.loads((tmp_path / "assets" / "final_assets_manifest.json").read_text(encoding="utf-8"))
    assert manifest["derived"]["task6_low_contrast"]["parent"] == "rock_original.jpg"


def test_approved_defaults_and_safe_synthetic_fallback():
    source = (prep.ROOT / "app.py").read_text(encoding="utf-8")
    defaults = (
        "poppies_rgb_grayscale.jpg",
        "dark_floral_tones.jpg",
        "sunset_solarization.jpg",
        "blue_cast_compass.jpg",
        "rock_low_contrast.png",
        "museum_clean.jpg",
    )
    assert "def preferred_sample(" in source
    assert all(name in source for name in defaults)
    assert "return approved if approved in iu.list_samples() else synthetic_fallback" in source
