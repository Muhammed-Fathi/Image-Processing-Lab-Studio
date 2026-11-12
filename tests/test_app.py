"""Application-level checks for the Streamlit app.

* Module import check (src package + app.py itself).
* requirements.txt completeness check.
* Startup behavior: every page of the app is executed through
  streamlit.testing.v1.AppTest and must run without exceptions.
"""

import importlib
import os
import sys

import pytest

from src import image_utils as iu

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_PATH = os.path.join(ROOT, "app.py")

ALL_PAGES = ["dashboard", "t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9"]


def app_module():
    """Import (once) the app module in bare mode; its NAV is the source of
    truth for the sidebar labels used to drive AppTest."""
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    return importlib.import_module("app")


def test_src_modules_import():
    from src import image_operations, image_utils, metrics  # noqa: F401


def test_app_module_imports_without_exception():
    """Importing app.py outside the Streamlit runtime must not raise
    (all Streamlit calls degrade to no-ops in bare mode)."""
    app_module()


def test_nav_covers_all_tasks():
    keys = [key for key, _ in app_module().NAV]
    assert keys == ["dashboard"] + ALL_PAGES[1:]


def test_requirements_complete():
    """requirements.txt must list every third-party module the project uses."""
    req_path = os.path.join(ROOT, "requirements.txt")
    with open(req_path, encoding="utf-8") as f:
        text = f.read()
    for dep in ("streamlit", "opencv-python-headless", "numpy", "matplotlib", "pillow"):
        assert dep in text, f"{dep} missing from requirements.txt"


def test_readme_and_gitignore_exist():
    assert os.path.isfile(os.path.join(ROOT, "README.md"))
    assert os.path.isfile(os.path.join(ROOT, ".gitignore"))


@pytest.mark.parametrize("page", ALL_PAGES)
def test_app_page_renders_without_exception(page):
    from streamlit.testing.v1 import AppTest

    label = dict(app_module().NAV)[page]
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.session_state["nav"] = label
    at.run()
    assert not at.exception, f"page '{page}' raised: {at.exception}"
    # The sidebar radio widget must be present on every page.
    assert len(at.sidebar.radio) == 1


@pytest.mark.parametrize("page", ALL_PAGES)
def test_app_page_shows_images(page):
    """Every task page must render at least one image (comparison or result);
    the dashboard renders none, which is expected."""
    from streamlit.testing.v1 import AppTest

    label = dict(app_module().NAV)[page]
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.session_state["nav"] = label
    at.run()
    assert not at.exception
    if page == "dashboard":
        assert len(at.main.image) == 0
    else:
        assert len(at.main.image) >= 1, f"page '{page}' shows no image"


# ---------------------------------------------------------------------------
# REGRESSION (audit issue 6): dashboard quick navigation must actually work
# ---------------------------------------------------------------------------

def test_dashboard_quick_navigation_switches_task():
    """Real widget interaction: pick a task in the dashboard quick-jump
    selectbox and verify the sidebar navigation ends up on that task."""
    from streamlit.testing.v1 import AppTest

    target = dict(app_module().NAV)["t6"]  # "Lab 3 · Task 6 — Histogram Enhancement"
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()  # initial dashboard render
    assert not at.exception

    jump_boxes = [s for s in at.main.selectbox if "Quick navigation" in (s.label or "")]
    assert len(jump_boxes) == 1
    jump_boxes[0].select(target)
    at.run()  # dashboard rerun queues pending_nav + st.rerun()

    assert not at.exception
    assert at.sidebar.radio[0].value == target
    # and the task page really rendered (its images are present)
    assert len(at.main.image) >= 1


# ---------------------------------------------------------------------------
# REGRESSION (audit issue 9): Task 4 affected-percentage must match the mask
# ---------------------------------------------------------------------------

def test_task4_solarization_percentage_matches_channel_mask():
    import numpy as np
    from streamlit.testing.v1 import AppTest

    img = iu.load_image_from_path(iu.sample_path("midtone_gradient.png"))
    threshold = 127  # the solarization slider default
    expected_pct = float(np.mean(np.asarray(img, dtype=np.uint8) >= threshold)) * 100
    expected_text = f"{expected_pct:.1f}%"

    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.session_state["nav"] = dict(app_module().NAV)["t4"]
    at.run()
    # switch the technique radio to Solarization and re-run
    at.main.radio[0].set_value("Solarization (partial inversion)")
    at.run()
    assert not at.exception
    found = any(expected_text in (md.value or "") for md in at.main.markdown)
    assert found, f"Task 4 does not show the per-channel mask percentage {expected_text}"


# ---------------------------------------------------------------------------
# REGRESSION (audit issue 2): show_image color-order contract
# ---------------------------------------------------------------------------

def test_show_image_color_order_contract(monkeypatch):
    import numpy as np

    captured = {}

    def fake_image(data, *args, **kwargs):
        captured["data"] = data
        captured["kwargs"] = kwargs

    monkeypatch.setattr(app_module().st, "image", fake_image)

    pure_red_rgb = np.array([[[255, 0, 0]]], dtype=np.uint8)
    pure_red_bgr = np.array([[[0, 0, 255]]], dtype=np.uint8)

    # order="rgb": the array must reach st.image unchanged (st.image is RGB)
    app_module().show_image(pure_red_rgb, order="rgb")
    assert captured["data"][0, 0].tolist() == [255, 0, 0]

    # order="bgr" (default): a BGR red pixel must be converted to RGB first
    app_module().show_image(pure_red_bgr, order="bgr")
    assert captured["data"][0, 0].tolist() == [255, 0, 0]

    # 2-D grayscale: passed through with the grayscale channel hint
    gray = np.array([[10, 200]], dtype=np.uint8)
    app_module().show_image(gray)
    assert captured["data"].tolist() == [[10, 200]]
    assert captured["kwargs"].get("channels") == "grayscale"


# ---------------------------------------------------------------------------
# REGRESSION (audit issue 3): exported PNGs must have correct colors
# ---------------------------------------------------------------------------

def test_download_button_export_preserves_rgb_colors(monkeypatch):
    import cv2
    import numpy as np

    captured = {}

    def fake_download(label, data, *args, **kwargs):
        # st.download_button(label, data, ...): the PNG bytes are the 2nd arg
        captured["data"] = data

    monkeypatch.setattr(app_module().st, "download_button", fake_download)

    def decoded_bgr(data):
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    pure_red_rgb = np.array([[[255, 0, 0]]], dtype=np.uint8)
    pure_blue_rgb = np.array([[[0, 0, 255]]], dtype=np.uint8)

    # RGB array with order="rgb" -> the exported PNG must decode (as BGR)
    # to (0, 0, 255) for red and (255, 0, 0) for blue.
    app_module().download_button(pure_red_rgb, "red.png", order="rgb")
    assert decoded_bgr(captured["data"])[0, 0].tolist() == [0, 0, 255]
    app_module().download_button(pure_blue_rgb, "blue.png", order="rgb")
    assert decoded_bgr(captured["data"])[0, 0].tolist() == [255, 0, 0]

    # BGR array with the default order: the red-blue positions are the
    # mirror image, proving the order contract is real (not a no-op).
    app_module().download_button(pure_red_rgb, "as_bgr.png")  # [255,0,0] is BGR blue
    assert decoded_bgr(captured["data"])[0, 0].tolist() == [255, 0, 0]


def test_encode_png_roundtrip_pure_colors():
    import cv2
    import numpy as np

    rgb_red = np.array([[[255, 0, 0]]], dtype=np.uint8)
    bgr = cv2.cvtColor(rgb_red, cv2.COLOR_RGB2BGR)
    data = iu.encode_png(bgr)
    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert decoded[0, 0].tolist() == [0, 0, 255]  # pure red, decoded as BGR


# ---------------------------------------------------------------------------
# REGRESSION (audit issue 10): Sobel explanation shows the standard kernels
# ---------------------------------------------------------------------------

def test_sobel_explanation_uses_standard_kernels():
    text = app_module().SOBEL_KX_STR
    # Standard 3x3 Sobel X (possibly negated) and Sobel Y rows must appear
    assert "[ -1   0   1 ]" in text
    assert "[ -2   0   2 ]" in text
    assert "[ -1  -2  -1 ]" in text
    assert "[  1   2   1 ]" in text
    # The old, non-standard [-2,0,2 / -1,0,1 / 2,0,-2] kernel must be gone
    assert "[ -2   0   2 ]\n  [ -1   0   1 ]\n  [  2   0  -2 ]" not in text
