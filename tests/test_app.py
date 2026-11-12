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
