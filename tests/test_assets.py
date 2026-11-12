"""Verify the bundled demonstration images exist and are usable.

If the samples are missing, run first:  python scripts/generate_samples.py
"""

import numpy as np

from src import image_utils as iu

EXPECTED_SAMPLES = [
    "color_scene.png",
    "dark_scene.png",
    "midtone_gradient.png",
    "color_cast.png",
    "low_contrast.png",
    "detail_scene.png",
    "geometric_edges.png",
]


def test_all_expected_samples_exist_and_load():
    for name in EXPECTED_SAMPLES:
        img = iu.load_image_from_path(iu.sample_path(name))
        assert img is not None, f"sample {name} missing or unreadable"
        assert img.ndim in (2, 3)
        assert img.dtype == np.uint8
        assert img.shape[0] > 0 and img.shape[1] > 0


def test_sample_registry_covers_all_bundled_files():
    for name in iu.list_samples():
        assert name in iu.SAMPLE_INFO, f"{name} has no entry in SAMPLE_INFO"


def test_samples_are_not_constant_images():
    # A constant demo image would make several tasks meaningless.
    for name in EXPECTED_SAMPLES:
        img = iu.load_image_from_path(iu.sample_path(name))
        assert int(img.max()) > int(img.min()), f"{name} is constant"


def test_dark_sample_is_actually_dark():
    scene = iu.load_image_from_path(iu.sample_path("color_scene.png"))
    dark = iu.load_image_from_path(iu.sample_path("dark_scene.png"))
    assert float(dark.mean()) < 0.5 * float(scene.mean())


def test_low_contrast_sample_has_narrow_range():
    img = iu.load_image_from_path(iu.sample_path("low_contrast.png"))
    rng = int(img.max()) - int(img.min())
    assert rng < 80, f"low_contrast sample range {rng} is not narrow"


def test_output_images_can_be_encoded_like_downloads():
    """The app's download buttons use encode_png; verify it works on a real sample."""
    img = iu.load_image_from_path(iu.sample_path("geometric_edges.png"))
    data = iu.encode_png(img)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(data) > 100  # a non-trivial image
