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
    """Use the actual image and threshold selected in Streamlit.

    The app may default to the final sunset photograph or to the original
    synthetic fallback. The regression must work for both.
    """
    import numpy as np
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.session_state["nav"] = dict(app_module().NAV)["t4"]
    at.run()

    technique = next(r for r in at.main.radio if r.label == "Technique")
    if technique.value != "Solarization (partial inversion)":
        technique.set_value("Solarization (partial inversion)")
        at.run()
    assert not at.exception

    selected_photo = next(
        item.value for item in at.main.selectbox
        if item.label == "Demonstration image"
    )
    img = iu.load_image_from_path(iu.sample_path(selected_photo))
    assert img is not None

    def assert_percentage_matches():
        threshold_slider = next(
            slider for slider in at.main.slider
            if slider.label.startswith("Solarization threshold T")
        )
        threshold = threshold_slider.value
        expected_pct = float(
            np.mean(np.asarray(img, dtype=np.uint8) >= threshold)
        ) * 100
        expected_text = f"{expected_pct:.1f}%"
        assert any(
            expected_text in (md.value or "") for md in at.main.markdown
        ), (
            f"Task 4 must report {expected_text} for "
            f"{selected_photo} at threshold {threshold}"
        )
        return threshold_slider

    slider = assert_percentage_matches()
    slider.set_value(200)
    at.run()
    assert not at.exception
    assert_percentage_matches()


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


def test_dashboard_ui_is_clean_and_has_no_raw_markdown_cards():
    """Guard the professional dashboard against emoji and literal ** markers."""
    import re
    source = open(APP_PATH, encoding="utf-8").read()
    assert not re.search(r"[\U0001F300-\U0001FAFF]", source)
    assert "**Task 1**" not in source
    assert 'class="lab-card"' in source
    assert "Quick navigation — select a module" in source
    assert source.count('class="task-eyebrow"') == 9



def test_rgb_matrix_excerpt_is_a_compact_spatial_grid(monkeypatch):
    """The 5x5 excerpt should display five image rows, not 25 flat records."""
    import numpy as np

    captured = {}
    monkeypatch.setattr(
        app_module().st, "table", lambda data, **kwargs: captured.update(data=data)
    )
    rgb = np.zeros((7, 9, 3), dtype=np.uint8)
    rgb[3, 4] = [12, 34, 56]
    app_module().matrix_excerpt_table(rgb, size=5, center=(3, 4))
    rows = captured["data"]

    assert len(rows) == 5
    assert len(rows[0]) == 6  # one row label + five coordinate columns
    assert rows[2]["4"] == "(12, 34, 56)"
    assert rows[0]["Row / Col"] == "1"


def test_download_button_allows_specific_channel_label(monkeypatch):
    """Download buttons should identify which output is being saved."""
    import numpy as np

    captured = {}
    def fake_download(label, data, *args, **kwargs):
        captured["label"] = label
        captured["data"] = data

    monkeypatch.setattr(app_module().st, "download_button", fake_download)
    app_module().download_button(
        np.zeros((2, 2, 3), dtype=np.uint8),
        "red.png",
        order="rgb",
        label="Download Red",
    )
    assert captured["label"] == "Download Red"
    assert captured["data"].startswith(b"\x89PNG")


# ---------------------------------------------------------------------------
# REGRESSION: concise sample menu and Task 7 controls
# ---------------------------------------------------------------------------

def test_sample_picker_keeps_description_outside_options():
    """Long sample descriptions must not overflow the shared dropdown."""
    source = open(APP_PATH, encoding="utf-8").read()
    assert "format_func=lambda s: s," in source
    assert 'st.caption(f"Sample details: {description}")' in source


def test_task7_noise_selector_gaussian():
    """Check that switching Task 7 to Gaussian noise still renders SNR."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.session_state["nav"] = dict(app_module().NAV)["t7"]
    at.run()
    assert not at.exception

    noise_radios = [r for r in at.main.radio if r.label == "Noise type"]
    assert len(noise_radios) == 1
    noise_radios[0].set_value("Gaussian (additive) noise")
    at.run()
    assert not at.exception
    assert any("SNR" in (m.label or "") for m in at.main.metric)


def test_task7_uses_clear_download_labels():
    source = open(APP_PATH, encoding="utf-8").read()
    task7 = source.split("def page_t7() -> None:", 1)[1].split(
        "# Task 8", 1
    )[0]
    assert 'label="Download Noisy Image"' in task7
    assert 'label="Download Denoised Image"' in task7
    assert "delta=improvement" in task7


# ---------------------------------------------------------------------------
# UI REGRESSION: consistent final layouts and distinct download labels
# ---------------------------------------------------------------------------

def test_all_task_pages_have_clear_download_labels():
    """No task page should leave outputs ambiguous as 'Download PNG'."""
    source = open(APP_PATH, encoding="utf-8").read()
    for task in range(1, 10):
        start = source.index(f"def page_t{task}() -> None:")
        if task < 9:
            end = source.index(f"# Task {task + 1}", start)
        else:
            end = source.index("# Main", start)
        section = source[start:end]
        assert "download_button(" in section, f"Task {task} missing download"
        for line in section.splitlines():
            if "download_button(" in line:
                # Task 1 passes the label in a multiline call; skip its
                # individual lines and check the relevant label expression.
                if task == 1:
                    continue
                assert "label=" in line, f"Task {task} has an ambiguous download"
    assert 'label=f"Download {channel_name}"' in source


def test_task5_original_not_rendered_twice():
    source = open(APP_PATH, encoding="utf-8").read()
    task5 = source.split("def page_t5() -> None:", 1)[1].split(
        "# Task 6", 1
    )[0]
    assert task5.count('show_image(rgb, "Original (RGB)", order="rgb")') == 1
    assert 'with st.expander("How to interpret channel elimination")' in task5


# ---------------------------------------------------------------------------
# Task 1 extension: HSV segmentation remains separate from RGB isolation
# ---------------------------------------------------------------------------

def test_task1_hsv_color_splash_ui():
    """Streamlit renders the optional HSV panel and accepts slider changes."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.session_state["nav"] = dict(app_module().NAV)["t1"]
    at.run()
    assert not at.exception

    sliders = {widget.label: widget for widget in at.main.slider}
    saturation = "Minimum saturation (HSV S)"
    brightness = "Minimum brightness (HSV V)"
    assert saturation in sliders
    assert brightness in sliders
    assert any(
        m.label == "Pixels matching the red HSV ranges" for m in at.main.metric
    )

    sliders[saturation].set_value(150)
    sliders[brightness].set_value(120)
    at.run()
    assert not at.exception
    updated_sliders = {widget.label: widget.value for widget in at.main.slider}
    assert updated_sliders[saturation] == 150
    assert updated_sliders[brightness] == 120


def test_task1_hsv_is_an_extra_tab_not_a_replacement():
    """Guard the original RGB channel/isolation code and HSV PNG exports."""
    source = open(APP_PATH, encoding="utf-8").read()
    task1 = source.split("def page_t1() -> None:", 1)[1].split(
        "# Task 2", 1
    )[0]
    assert '"Color isolation"' in task1
    assert '"HSV Red Color Splash"' in task1
    assert 'ops.isolate_channel(rgb, index)' in task1
    assert 'ops.hsv_red_color_splash(' in task1
    assert 'label="Download HSV Red Mask"' in task1
    assert 'label="Download HSV Color Splash"' in task1
