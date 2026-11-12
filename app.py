"""Image Processing Lab Studio - main Streamlit application.

A single educational app covering the nine university image-processing
tasks across four labs. All algorithms live in `src/` as plain NumPy /
OpenCV functions; this file only builds the interactive UI around them.
"""

from __future__ import annotations

import os

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src import image_operations as ops
from src import image_utils as iu
from src import metrics

matplotlib.use("Agg", force=False)

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Image Processing Lab Studio",
    page_icon="I",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root { --ink:#193449; --muted:#6c7c8c; --accent:#176e82; --line:#dfe7ed; }
.stApp { background:#f7f9fb; }
.block-container { max-width:1290px; padding-top:2rem; padding-bottom:3rem; }
header[data-testid="stHeader"] { background:transparent; }
section[data-testid="stSidebar"] { background:#eef3f7; border-right:1px solid var(--line); }
h1,h2,h3 { color:var(--ink); letter-spacing:-.025em; }
.sidebar-brand { padding:8px 2px 20px; }
.sidebar-brand small { display:block; color:var(--accent); font-size:11px; font-weight:800; letter-spacing:.15em; }
.sidebar-brand h2 { font-size:27px; font-weight:760; margin:7px 0 4px; }
.sidebar-brand p { color:var(--muted); font-size:13px; margin:0; }
.sidebar-label { color:var(--muted); font-size:11px; font-weight:800; letter-spacing:.12em; margin:8px 0 14px; }
.hero { border-radius:18px; padding:42px 45px; background:linear-gradient(120deg,#142e43,#1a5365); color:#fff; margin:6px 0 26px; }
.hero .overline { color:#a6dbe0; font-size:11px; letter-spacing:.14em; font-weight:800; margin-bottom:12px; }
.hero h1 { color:#fff; font-size:clamp(29px,3.5vw,45px); margin:0 0 13px; font-weight:760; line-height:1.12; }
.hero p { color:#d6e3ea; font-size:15px; line-height:1.7; max-width:790px; margin:0; }
.hero .chips { display:flex; flex-wrap:wrap; gap:8px; margin-top:23px; }
.hero .chips span { font-size:11px; font-weight:700; color:#eaf4f6; padding:6px 12px; border:1px solid #628895; border-radius:30px; }
.section-heading { margin:28px 0 14px; }
.section-heading h2 { font-size:23px; margin:0 0 6px; }
.section-heading p { font-size:14px; color:var(--muted); margin:0; }
.lab-card { min-height:239px; background:#fff; border:1px solid var(--line); border-radius:14px; padding:21px 23px; margin-bottom:15px; box-shadow:0 4px 14px rgba(26,56,74,.035); }
.lab-card .meta { display:flex; justify-content:space-between; gap:9px; color:var(--accent); font-size:11px; font-weight:800; letter-spacing:.08em; margin-bottom:15px; }
.lab-card .count { color:#81919e; font-weight:600; letter-spacing:0; }
.lab-card h3 { font-size:19px; font-weight:750; margin:0 0 14px; }
.lab-card .task-row { display:flex; gap:12px; border-top:1px solid #ecf0f4; padding:9px 0; font-size:14px; color:#284255; }
.lab-card .task-no { color:var(--accent); font-size:12px; font-weight:800; min-width:20px; }
.task-eyebrow { color:var(--accent); font-size:11px; letter-spacing:.12em; font-weight:800; margin:7px 0 6px; }
.workflow { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:17px 0 28px; }
.workflow div { background:#fff; border:1px solid var(--line); border-radius:12px; padding:16px; }
.workflow strong { display:block; font-size:11px; letter-spacing:.1em; color:var(--accent); margin-bottom:7px; }
.workflow span { font-size:13px; color:#344e60; line-height:1.5; }
.source-heading { color:var(--muted); font-size:11px; font-weight:800; letter-spacing:.12em; margin:12px 0 8px; }
div[data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-radius:12px; padding:13px; }
@media(max-width:820px) { .hero { padding:27px 22px; } .workflow { grid-template-columns:repeat(2,1fr); } .lab-card { min-height:auto; } }
</style>
""", unsafe_allow_html=True)

NAV: list[tuple[str, str]] = [
    ("dashboard", "Overview"),
    ("t1", "01  RGB Channels"),
    ("t2", "02  Weighted Grayscale"),
    ("t3", "03  Arithmetic Operations"),
    ("t4", "04  Negative / Solarization"),
    ("t5", "05  Channel Elimination"),
    ("t6", "06  Histogram Enhancement"),
    ("t7", "07  Noise Reduction + SNR"),
    ("t8", "08  Image Sharpening"),
    ("t9", "09  Edge Detection"),
]

NAV_BY_LABEL = {label: key for key, label in NAV}

ACCEPTED_FORMATS = ["png", "jpg", "jpeg", "bmp", "tiff", "tif", "webp"]


# ---------------------------------------------------------------------------
# Shared UI helpers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_sample_cached(name: str) -> np.ndarray | None:
    """Load a bundled sample image (cached across reruns)."""
    return iu.load_image_from_path(iu.sample_path(name))


def image_source(default_sample: str, color_preferred: bool = True):
    """Shared 'where does the image come from' block.

    The user either picks a bundled demonstration image or uploads their
    own. Returns (image, source_label) with the image as a uint8 array
    (BGR color or 2-D gray).
    """
    st.markdown('<div class="source-heading">IMAGE SOURCE</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([3, 2])
    samples = iu.list_samples()
    with c1:
        if not samples:
            st.warning("No sample images found in assets/samples — run `python scripts/generate_samples.py`.")
            choice = None
        else:
            idx = samples.index(default_sample) if default_sample in samples else 0
            choice = st.selectbox(
                "Demonstration image",
                samples,
                index=idx,
                format_func=lambda s: s,
                label_visibility="visible",
            )
    with c2:
        up = st.file_uploader(
            "Upload a custom image",
            type=ACCEPTED_FORMATS,
            label_visibility="visible",
        )

    if up is None and choice is not None:
        description = iu.SAMPLE_INFO.get(choice)
        if description:
            st.caption(f"Sample details: {description}")

    if up is not None:
        data = up.getvalue()
        if not iu.is_valid_image_bytes(data):
            st.error("That file is not a readable image. Supported formats: " + ", ".join(ACCEPTED_FORMATS))
            st.stop()
        img = iu.load_image_from_bytes(data)
        if img is None:
            st.error("Could not decode the uploaded image file.")
            st.stop()
        if color_preferred and img.ndim == 2:
            st.info("Your upload is grayscale; it will be converted to 3 channels where a color image is needed.")
        return img, f"Uploaded: {up.name}"

    if choice is None:
        st.error("Please upload an image to continue.")
        st.stop()
    img = load_sample_cached(choice)
    if img is None:
        st.error(f"Could not read the sample image '{choice}'.")
        st.stop()
    if color_preferred and img.ndim == 2:
        st.info("The selected sample is grayscale; it will be converted to 3 channels where a color image is needed.")
    return img, f"Sample: {choice}"


def source_basename(source_label: str) -> str:
    """Short file name from a source label, for download file names."""
    name = source_label.split(": ", 1)[-1]
    return os.path.splitext(name)[0][:40] or "image"


def show_image(img: np.ndarray, caption: str | None = None, order: str = "bgr") -> None:
    """Display an image with the correct color order.

    `order` declares the channel order of the input array: "bgr" (OpenCV
    default, converted before display) or "rgb" (already display-ready).
    2-D grayscale arrays are unaffected by `order`.
    """
    if img.ndim == 2:
        st.image(img, channels="grayscale", caption=caption)
    elif order == "rgb":
        st.image(img, caption=caption)
    else:
        st.image(ops.bgr_to_rgb(img), caption=caption)


def download_button(img: np.ndarray, name: str, order: str = "bgr", label: str = "Download PNG") -> None:
    """Offer a PNG download.

    cv2.imencode assumes the array is in BGR order, so RGB arrays are
    converted before encoding - otherwise the exported file would have
    swapped red and blue channels.
    """
    arr = img if order == "bgr" else cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    st.download_button(
        label,
        iu.encode_png(arr),
        file_name=name,
        mime="image/png",
        use_container_width=True,
    )


def formula_box(text: str, title: str = "Formula / mathematical rule") -> None:
    with st.expander(title):
        st.code(text, language=None)


def discussion(points: list[str], title: str = "Discussion notes (for oral evaluation)") -> None:
    with st.expander(title, expanded=False):
        for p in points:
            st.markdown(f"- {p}")
        st.caption("These notes summarize what to say if a teaching assistant asks for the reasoning behind the choices on this page.")


def stats_table(rows: list[dict], title: str = "Intensity statistics") -> None:
    st.markdown(f"**{title}**")
    st.table(rows)


def format_stats(s: dict) -> dict:
    return {
        "Mean": f"{s['mean']:.2f}",
        "Std": f"{s['std']:.2f}",
        "Min": f"{s['min']:.0f}",
        "Max": f"{s['max']:.0f}",
    }


def histogram_figure(gray_images: dict[str, np.ndarray], title: str = "Histogram") -> plt.Figure:
    """Matplotlib overlay of the intensity histograms of one or more
    gray images, computed from the actual pixel data (256 bins)."""
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for label, g in gray_images.items():
        counts, _ = metrics.histogram(np.asarray(g, dtype=np.uint8))
        ax.plot(np.arange(256), counts, label=label, linewidth=1.3)
    ax.set_xlabel("Intensity (0–255)")
    ax.set_ylabel("Pixel count")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def matrix_excerpt_table(rgb: np.ndarray, size: int = 5,
                         center: tuple[int, int] | None = None) -> None:
    """Display an actual size-by-size matrix of RGB tuples, with one row per image row.

    Center is a (row, column) pair. Unlike a 25-row pixel list, this reflects
    the two-dimensional spatial structure of the sampled image region.
    """
    h, w = rgb.shape[:2]
    if center is None:
        center = (h // 2, w // 2)
    cy, cx = center
    y0 = min(max(0, cy - size // 2), max(0, h - size))
    x0 = min(max(0, cx - size // 2), max(0, w - size))
    xs = list(range(x0, min(w, x0 + size)))
    ys = list(range(y0, min(h, y0 + size)))
    rows = []
    for y in ys:
        row = {"Row / Col": str(y)}
        for x in xs:
            r, g, b = (int(value) for value in rgb[y, x])
            row[str(x)] = f"({r}, {g}, {b})"
        rows.append(row)
    st.table(rows)

def luminance(img: np.ndarray) -> np.ndarray:
    """Grayscale (or Y-channel) version used for histogram display."""
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

def render_sidebar() -> str:
    """Keep the navigation short; apply queued dashboard jumps before radio creation."""
    pending = st.session_state.pop("pending_nav", None)
    if pending in NAV_BY_LABEL:
        st.session_state["nav"] = pending
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-brand"><small>IMAGE PROCESSING</small>'
            '<h2>Lab Studio</h2><p>Interactive practical workspace</p></div>'
            '<div class="sidebar-label">NAVIGATION</div>',
            unsafe_allow_html=True,
        )
        labels = [label for _, label in NAV]
        st.radio("Navigate", labels, index=0, key="nav", label_visibility="collapsed")
        st.divider()
        st.caption("4 lab groups / 9 practical modules")
        st.caption("Select a sample image or upload your own inside any module.")
    nav = st.session_state.get("nav")
    if nav not in NAV_BY_LABEL:
        nav = NAV[0][1]
    return NAV_BY_LABEL[nav]

# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

def page_dashboard() -> None:
    st.markdown(
        '<section class="hero"><div class="overline">COMPUTER VISION / UNIVERSITY LAB</div>'
        '<h1>Image Processing Lab Studio</h1>'
        '<p>Explore classical image-processing techniques in one focused workspace. '
        'Choose a module, adjust parameters, compare results and understand the mathematics.</p>'
        '<div class="chips"><span>04 LAB GROUPS</span><span>09 MODULES</span>'
        '<span>PYTHON + OPENCV</span></div></section>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-heading"><h2>Explore the labs</h2>'
        '<p>Nine practical exercises, grouped by topic.</p></div>',
        unsafe_allow_html=True,
    )
    cards = [
        ("01", "Image Fundamentals", [
            ("01", "RGB Channel Separation"),
            ("02", "Custom Weighted Grayscale"),
        ]),
        ("02", "Pixel & Intensity Operations", [
            ("03", "Arithmetic Enhancement"),
            ("04", "Negative and Solarization"),
            ("05", "Color Channel Elimination"),
        ]),
        ("03", "Histogram Enhancement", [
            ("06", "Histogram Stretching and Equalization"),
        ]),
        ("04–05", "Filtering & Edge Detection", [
            ("07", "Noise Reduction and SNR"),
            ("08", "Blur and Sharpening"),
            ("09", "Edge Detection"),
        ]),
    ]
    for start in range(0, len(cards), 2):
        cols = st.columns(2, gap="medium")
        for col, (lab_no, title, tasks) in zip(cols, cards[start:start + 2]):
            with col:
                rows = "".join(
                    f'<div class="task-row"><span class="task-no">{num}</span>'
                    f'<span>{name}</span></div>' for num, name in tasks
                )
                st.markdown(
                    f'<section class="lab-card"><div class="meta"><span>LAB {lab_no}</span>'
                    f'<span class="count">{len(tasks)} MODULES</span></div>'
                    f'<h3>{title}</h3>{rows}</section>',
                    unsafe_allow_html=True,
                )

    st.markdown(
        '<div class="section-heading"><h2>Open a module</h2>'
        '<p>Use the sidebar or jump directly to an exercise.</p></div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.5, 1], gap="large")
    with left:
        jump = st.selectbox(
            "Quick navigation — select a module",
            ["Select a module"] + [label for _, label in NAV[1:]],
        )
        if jump != "Select a module" and st.session_state.get("pending_nav") != jump:
            st.session_state["pending_nav"] = jump
            st.rerun()
    with right:
        st.info("Every module includes sample images, controls, a before/after comparison, and discussion notes.")

    st.markdown(
        '<div class="section-heading"><h2>How it works</h2>'
        '<p>A simple workflow for every practical exercise.</p></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="workflow">'
        '<div><strong>01 / SELECT</strong><span>Choose a module.</span></div>'
        '<div><strong>02 / INPUT</strong><span>Load a sample or upload an image.</span></div>'
        '<div><strong>03 / PROCESS</strong><span>Adjust settings and compare results.</span></div>'
        '<div><strong>04 / REVIEW</strong><span>Inspect formulas and download.</span></div>'
        '</div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Task 1 — RGB channel separation
# ---------------------------------------------------------------------------

def page_t1() -> None:
    st.markdown('<div class="task-eyebrow">LAB 01 / TASK 01</div>', unsafe_allow_html=True)
    st.title("RGB Channel Separation")
    st.caption(
        "Separate the red, green, and blue channels with OpenCV and "
        "inspect their intensity, color contribution, and pixel values."
    )

    img, source = image_source("color_scene.png")
    rgb = ops.bgr_to_rgb(img)
    red, green, blue = ops.separate_channels(rgb)
    h, w = rgb.shape[:2]

    # A compact overview makes the output accessible without scrolling past
    # two large copies of the same image.
    st.markdown("### Original image")
    left, right = st.columns([1.55, 1], gap="large")
    with left:
        show_image(rgb, "Original image — correct RGB color order", order="rgb")
    with right:
        st.markdown("**Method overview**")
        st.markdown(
            "OpenCV reads color images as **BGR**. Convert to **RGB** before "
            "interpreting individual red, green and blue channels."
        )
        st.code(
            "rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)\n"
            "red, green, blue = cv2.split(rgb)",
            language="python",
        )
        st.metric("Resolution", f"{w} × {h}")
        st.caption("Each channel has 8-bit intensity values from 0 to 255.")

    # BGR/RGB is useful for an oral exam, but shouldn't dominate the page.
    with st.expander("Why does BGR versus RGB matter? See the comparison"):
        st.markdown(
            "OpenCV's **BGR** channel order differs from the **RGB** order "
            "used for display. Treating BGR as RGB swaps the red and blue appearance."
        )
        before, after = st.columns(2)
        with before:
            st.image(
                img,
                caption="Incorrect: raw BGR interpreted as RGB (red and blue swapped)",
            )
        with after:
            st.image(
                rgb,
                caption="Correct: BGR converted to RGB before display",
            )

    st.markdown("### Explore the channels")
    tab_map, tab_iso, tab_px, tab_mx = st.tabs(
        ["Intensity maps", "Color isolation", "Pixel inspector", "RGB matrix (5×5)"]
    )
    with tab_map:
        st.caption(
            "Brightness in each grayscale map represents the intensity "
            "of that channel at every pixel."
        )
        cols = st.columns(3)
        for col, values, name in zip(
            cols, (red, green, blue), ("Red", "Green", "Blue")
        ):
            with col:
                st.image(values, channels="grayscale", caption=f"{name} intensity")

    with tab_iso:
        st.caption(
            "Each image preserves one color channel and sets the other two to zero."
        )
        cols = st.columns(3)
        base = source_basename(source)
        for col, index, channel_name, image_channel in zip(
            cols,
            (0, 1, 2),
            ("Red", "Green", "Blue"),
            ("red", "green", "blue"),
        ):
            with col:
                isolated = ops.isolate_channel(rgb, index)
                st.image(isolated, caption=f"{channel_name} channel only")
                download_button(
                    isolated,
                    f"task1_{base}_{image_channel}_only.png",
                    order="rgb",
                    label=f"Download {channel_name}",
                )

    with tab_px:
        st.caption("Choose pixel coordinates to inspect the original RGB values.")
        controls, preview = st.columns([1, 1.35], gap="large")
        with controls:
            px = st.slider("Column x", 0, max(0, w - 1), min(50, w - 1))
            py = st.slider("Row y", 0, max(0, h - 1), min(50, h - 1))
            r_v, g_v, b_v = (int(v) for v in rgb[py, px])
            c1, c2, c3 = st.columns(3)
            c1.metric("Red", r_v)
            c2.metric("Green", g_v)
            c3.metric("Blue", b_v)
            st.caption(f"Selected pixel: row {py}, column {px}")
        with preview:
            marked = rgb.copy()
            cv2.circle(marked, (px, py), 4, (255, 0, 0), 2)
            st.image(marked, caption="Pixel location in the original image")

    with tab_mx:
        st.caption(
            "This is a 5×5 spatial matrix of (R, G, B) pixel triplets. "
            "Move the coordinates toward an object boundary to see how values change."
        )
        cy_col, cx_col = st.columns(2)
        with cy_col:
            matrix_y = st.slider("Matrix center — row", 0, max(0, h - 1), h // 2)
        with cx_col:
            matrix_x = st.slider("Matrix center — column", 0, max(0, w - 1), w // 2)
        matrix_excerpt_table(rgb, size=5, center=(matrix_y, matrix_x))

    with st.expander("Understanding the result and mathematical operations"):
        st.markdown(
            "**RGB representation:** each pixel contains three channel values "
            "between 0 and 255. Grayscale channel intensity maps show the strength "
            "of each primary color. Color-isolated images instead keep the actual "
            "primary in the rendered output."
        )
        st.code(
            "Pixel = (R, G, B), each component in [0, 255]\n"
            "rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)\n"
            "R, G, B = cv2.split(rgb)\n"
            "isolate(red) = (R, 0, 0)\n"
            "isolate(green) = (0, G, 0)\n"
            "isolate(blue) = (0, 0, B)",
            language="python",
        )

    discussion([
        "OpenCV typically reads color images in BGR; convert to RGB before interpreting channels.",
        "An intensity map uses grayscale brightness to reveal the contribution of one color channel.",
        "A color-isolated image retains exactly one channel while setting the others to zero.",
        "The RGB split is reversible when all three channels are kept; no color information is lost by splitting alone.",
    ])

# ---------------------------------------------------------------------------
# Task 2 — custom weighted grayscale
# ---------------------------------------------------------------------------

def page_t2() -> None:
    st.markdown('<div class="task-eyebrow">LAB 01 / TASK 02</div>', unsafe_allow_html=True)
    st.title("Custom Grayscale Conversion")
    st.caption("Lab 1 — Image Fundamentals · Grayscale via the weighted luminance formula, implemented manually (no average, no cv2 conversion).")

    img, source = image_source("color_scene.png")
    rgb = ops.bgr_to_rgb(img)
    gray_custom = ops.weighted_grayscale(rgb)
    gray_cv = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)  # reference for comparison only
    max_diff = int(np.max(np.abs(gray_custom.astype(np.int16) - gray_cv.astype(np.int16))))

    st.markdown("#### Original vs. custom weighted grayscale")
    c1, c2, c3 = st.columns(3)
    with c1:
        show_image(rgb, "Original (RGB)", order="rgb")
    with c2:
        st.image(gray_custom, channels="grayscale", caption="This task's output: 0.299R + 0.587G + 0.114B")
    with c3:
        st.image(gray_cv, channels="grayscale",
                 caption="OpenCV's own grayscale — shown ONLY as a reference check, not used in the implementation")

    m1, m2 = st.columns(2)
    m1.metric("Max |custom − OpenCV|", f"{max_diff} / 255",
              "≤ 1 expected: same formula, only rounding differs")
    s = metrics.intensity_stats(gray_custom)
    m2.metric("Result mean intensity", f"{s['mean']:.1f}")

    h, w = rgb.shape[:2]
    st.markdown("#### Worked example on a selected pixel")
    c1, c2 = st.columns(2)
    px = c1.slider("Column x", 0, max(0, w - 1), min(50, max(0, w - 1)))
    py = c2.slider("Row y", 0, max(0, h - 1), min(50, max(0, h - 1)))
    R, G, B = (int(v) for v in rgb[py, px])
    tr, tg, tb = 0.299 * R, 0.587 * G, 0.114 * B
    total = tr + tg + tb
    avg = (R + G + B) / 3.0
    st.code(
        f"Pixel ({py}, {px}):  R = {R}   G = {G}   B = {B}\n\n"
        f"Gray = 0.299*R + 0.587*G + 0.114*B\n"
        f"     = 0.299*{R} + 0.587*{G} + 0.114*{B}\n"
        f"     = {tr:.2f} + {tg:.2f} + {tb:.2f}\n"
        f"     = {total:.2f}  ->  uint8 {int(min(255, max(0, total)))}\n\n"
        f"Simple average for comparison: (R+G+B)/3 = ({R}+{G}+{B})/3 = {avg:.2f}",
        language=None,
    )
    m1, m2 = st.columns(2)
    m1.metric("Weighted result", f"{total:.2f} → {int(min(255, max(0, total)))}")
    m2.metric("Simple average", f"{avg:.2f}",
              f"differs by {abs(total - avg):.2f} at this pixel")

    formula_box(
        "Gray = 0.299*R + 0.587*G + 0.114*B        (ITU-R BT.601 luminance)\n"
        "implementation: float32 -> np.clip(0, 255) -> uint8\n\n"
        "Why not the average?  (R+G+B)/3 treats the three primaries as equally\n"
        "visible to the human eye, which is false: under normal (photopic)\n"
        "lighting the eye is most sensitive to green, less to red, and least\n"
        "to blue. The weights 0.299 / 0.587 / 0.114 come from the human\n"
        "luminous-efficiency function, so green contributes the most —\n"
        "hence a green leaf stays bright in weighted grayscale while an\n"
        "equally bright blue patch is rendered darker.",
        "Formula and rationale",
    )

    st.markdown(
        """
**Why this technique?** The university rule is to convert to grayscale with a weighted
formula, not a plain average or median. The luminance weights match human perception, so
the gray image preserves the *perceived* brightness of the scene. The manual NumPy
implementation (float → clip → uint8) is deliberately shown because it is exactly what
the formula says, and the comparison column proves it matches OpenCV's own conversion up
to rounding. **Limitation:** grayscale always discards hue — two different colors with the
same luminance become identical gray values.
        """
    )
    discussion([
        "Green weight (0.587) is the largest because the eye's photopic sensitivity peaks near 555 nm (green).",
        "Weighted vs. averaged: they differ whenever the channel values are uneven — see the live example above.",
        "Float arithmetic before clipping avoids uint8 wrap-around; the result matches OpenCV within ±1 (rounding only).",
        "Limitation: luminance-only conversion loses chromatic information permanently.",
    ])

    st.markdown("#### Download")
    download_button(gray_custom, f"task2_{source_basename(source)}_weighted_gray.png")


# ---------------------------------------------------------------------------
# Task 3 — arithmetic operations
# ---------------------------------------------------------------------------

OP_INFO = {
    "add": ("Addition — I + C", "I_out = clip(I + C, 0, 255)"),
    "sub": ("Subtraction — I − C", "I_out = clip(I − C, 0, 255)"),
    "mul": ("Multiplication — I × C", "I_out = clip(I × C, 0, 255)"),
    "div": ("Division — I ÷ C", "I_out = clip(I ÷ C, 0, 255)   (C clamped away from 0)"),
}


def exposure_guidance(stats: dict) -> tuple[str, str]:
    m = stats["mean"]
    if m < 85:
        return "dark", (
            f"This image is **dark** (mean = {m:.1f} / 255). To lift brightness, *addition* "
            f"lifts every pixel by a constant (fast, but shadows can clip to white) and "
            f"*multiplication* scales the whole range (keeps relative contrast). Subtraction "
            f"or division would make it darker — choose them only if you want that."
        )
    if m > 170:
        return "bright", (
            f"This image is **bright / overexposed** (mean = {m:.1f} / 255). To bring it down, "
            f"*subtraction* lowers every pixel by a constant and *division* compresses the "
            f"whole range (highlights compress more). Addition or multiplication would make "
            f"it brighter — choose them only if you want that."
        )
    return "balanced", (
        f"This image is **roughly balanced** (mean = {m:.1f} / 255). Any operation works; "
        f"addition/subtraction shift the whole curve up or down, multiplication/division "
        f"scale it. Note: multiplication by C<1 and division by C>1 both darken, while "
        f"C>1 multiplies and C<1 divisions brighten."
    )


def page_t3() -> None:
    st.markdown('<div class="task-eyebrow">LAB 02 / TASK 03</div>', unsafe_allow_html=True)
    st.title("Arithmetic Image Enhancement")
    st.caption("Lab 2 — Pixel & Intensity Operations · Choose the arithmetic operation that fits the image's visual problem.")

    img, source = image_source("dark_scene.png")
    stats_in = metrics.intensity_stats(img)
    label, advice = exposure_guidance(stats_in)
    st.info(advice)

    c1, c2 = st.columns([1, 2])
    with c1:
        op = st.selectbox("Arithmetic operation", list(OP_INFO.keys()),
                          format_func=lambda k: OP_INFO[k][0])
    with c2:
        if op == "add":
            c_val = st.slider("Constant C", -128, 128, 30)
        elif op == "sub":
            c_val = st.slider("Constant C  (negative C brightens)", -128, 128, 30)
        elif op == "mul":
            c_val = st.slider("Multiplier C  (C<1 darkens, C>1 brightens)", 0.0, 3.0, 1.5, 0.05)
        else:
            c_val = st.slider("Divisor C  (C>1 darkens, 0<C<1 brightens)", 0.1, 5.0, 1.5, 0.05)

    out = ops.arithmetic_operation(img, op, c_val)
    stats_out = metrics.intensity_stats(out)

    st.markdown("#### Original vs. processed")
    c1, c2 = st.columns(2)
    with c1:
        show_image(img, "Original")
    with c2:
        show_image(out, f"Processed — {OP_INFO[op][0]} with C = {c_val}")

    stats_table(
        [
            {"Image": "Original", **format_stats(stats_in)},
            {"Image": f"Processed ({OP_INFO[op][0]})", **format_stats(stats_out)},
        ],
        "Intensity statistics (whole image)",
    )
    st.pyplot(histogram_figure({"Original": luminance(img), "Processed": luminance(out)},
                               "Intensity histogram before / after"))

    if op == "add":
        effect = (f"Every pixel was lifted by {c_val}. Dark regions become visible first; "
                  f"pixels above 255−{c_val} clip to white (detail lost there). " if c_val > 0 else
                  f"Every pixel was lowered by {abs(c_val)} (detail in the darks clips to black).")
        why = ("Addition is the classic choice for an **underexposed** image: it shifts the "
               "histogram to the right, lifting shadow content that is still above black. "
               "Values that were clipped to 0 at capture time cannot be brought back by any "
               "point operation — addition only makes the surviving dark content visible.")
    elif op == "sub":
        effect = (f"Every pixel was lowered by {c_val}. Overexposed highlights are pulled back; "
                  f"pixels below {c_val} clip to black." if c_val > 0 else
                  f"Every pixel was lifted by {abs(c_val)} (C is negative).")
        why = ("Subtraction is the classic choice for an **overexposed** image: it shifts the "
               "histogram to the left, pulling highlight content that is still below white "
               "back into range. Highlights already clipped to 255 at capture time cannot be "
               "restored by any point operation.")
    elif op == "mul":
        effect = (f"Every pixel was scaled by {c_val}. The whole histogram was stretched/compressed "
                  f"around zero; the darker the pixel, the smaller the absolute change.")
        why = ("Multiplication **scales** the dynamic range: C > 1 brightens highlights more than "
               "shadows, C < 1 darkens. Use it when the image is dark but you want to preserve "
               "relative contrast across the range.")
    else:
        effect = (f"Every pixel was divided by {c_val}. Values above 255/C would clip to white; "
                  f"the histogram is compressed towards zero for C > 1.")
        why = ("Division **compresses** the range: C > 1 darkens (highlights the most), 0 < C < 1 "
               "brightens. Useful for overexposed images where a constant shift would clip shadows.")

    formula_box(OP_INFO[op][1] + "\n\ncomputed in float32, then clipped to [0, 255] -> uint8, so unsigned overflow cannot occur")

    st.markdown(f"**Selected operation and why:** {why} {effect}")
    st.markdown(
        "**Honest note:** no arithmetic operation is universally best. Addition is cheap and "
        "predictable but clips; multiplication keeps relative contrast but overdrives "
        "highlights; subtraction/division are the mirror images. The right choice follows the "
        "image's measured exposure (see the banner above), and you should be able to justify "
        "yours with the statistics shown here."
    )
    discussion([
        f"The input's measured exposure is “{label}” (mean {stats_in['mean']:.1f}); the chosen operation {OP_INFO[op][0]} was selected to fight that problem.",
        "All four operations are affine/linear point operations — each pixel is processed independently.",
        "Float + clip protects against uint8 wrap-around (250+10 would wrap to 4 in raw uint8 math).",
        "Limitation: point operations cannot restore information that was clipped at capture time.",
    ])

    st.markdown("#### Download")
    download_button(out, f"task3_{source_basename(source)}_{op}_{str(c_val).replace('.', '_')}.png")


# ---------------------------------------------------------------------------
# Task 4 — negative / solarization
# ---------------------------------------------------------------------------

def page_t4() -> None:
    st.markdown('<div class="task-eyebrow">LAB 02 / TASK 04</div>', unsafe_allow_html=True)
    st.title("Complement (Negative) & Solarization")
    st.caption("Lab 2 — Pixel & Intensity Operations · Full tonal inversion vs. threshold-controlled partial inversion.")

    img, source = image_source("midtone_gradient.png")
    stats = metrics.intensity_stats(img)

    mode = st.radio(
        "Technique",
        ["Negative (complement)", "Solarization (partial inversion)"],
        horizontal=True,
    )
    threshold = 127
    if mode.startswith("Solarization"):
        threshold = st.slider("Solarization threshold T  (pixels with I ≥ T are inverted)", 0, 255, 127)
        out = ops.solarize(img, threshold)
    else:
        out = ops.negative(img)

    st.markdown("#### Original vs. processed")
    c1, c2 = st.columns(2)
    with c1:
        show_image(img, "Original")
    with c2:
        show_image(out, "Negative" if mode.startswith("Negative") else f"Solarized (T = {threshold})")

    if mode.startswith("Negative"):
        frac = None
        formula_box(
            "out = 255 - I     for every pixel (each channel)\n\n"
            "Tonal inversion: blacks -> whites, whites -> blacks, midtones mirrored.\n"
            "Useful: inspecting shadow detail (it moves into the bright range),\n"
            "X-ray-like medical look, artistic effects, and as a quick check of\n"
            "tonal distribution.",
            "Formula — negative (complement)",
        )
        chosen = (
            f"The **negative** was selected here. It is the simplest point transformation "
            f"(a perfect mirror of the tonal curve) and it is the right tool when the "
            f"*entire* image needs its tonal polarity flipped — for example to study what is "
            f"hidden in the dark areas of this image (mean {stats['mean']:.1f}): after "
            f"inversion those areas are bright and easy to see."
        )
    else:
        # The exact per-channel mask the solarize() rule acts on:
        frac = float(np.mean(np.asarray(img, dtype=np.uint8) >= threshold)) * 100
        formula_box(
            f"out = 255 - I   if I >= T   (T = {threshold})\n"
            f"out = I         if I <  T\n\n"
            "Partial inversion: only the bright part of the image is flipped, the dark part\n"
            "stays. Raising T inverts fewer pixels; lowering T inverts almost everything\n"
            "(T = 0 gives exactly the negative). Produces the classic solarized,\n"
            "half-tone-like look around the threshold.",
            "Formula — solarization",
        )
        chosen = (
            f"**Solarization with T = {threshold}** was selected. The rule inverts every "
            f"stored channel value with I ≥ T, and about **{frac:.1f}% of the image's stored "
            f"channel values** satisfy that (measured on the actual array, per channel) — the "
            f"highlights flip while the shadows stay put. That makes it the "
            f"right choice when you want to *emphasize or re-shape only the bright areas* "
            f"(e.g. blown-out skies, bright clothing, reflections) without disturbing the "
            f"rest of the picture, unlike the negative which changes everything."
        )

    st.markdown(f"**Why this technique was selected:** {chosen}")
    st.markdown(
        "**Visual situations where each is useful.** *Negative*: viewing shadow detail, "
        "X-ray aesthetics, checking whether an image is balanced (the negative often reveals "
        "problems). *Solarization*: dramatic highlight effects, halftone-style art, or "
        "selectively flattening overexposed regions.\n\n"
        "**Limitation:** both are point operations — they reorder tonal values but never "
        "recover lost detail. Solarization creates a visible band at the threshold (by "
        "design); the negative inverts *all* information, so it is rarely the final output."
    )
    discussion([
        "Negative = full mirror of the intensity axis: out = 255 − I.",
        "Solarization = the negative applied only above threshold T; T is the control knob (T=0 collapses to the full negative).",
        "The explanation shown above is generated from the actual image statistics, so it always matches what is displayed.",
        "Limitation: point operations change tone, not structure; threshold artifacts are inherent to solarization.",
    ])

    st.markdown("#### Download")
    tag = "negative" if mode.startswith("Negative") else f"solarize_T{threshold}"
    download_button(out, f"task4_{source_basename(source)}_{tag}.png")


# ---------------------------------------------------------------------------
# Task 5 — channel elimination
# ---------------------------------------------------------------------------

CHANNEL_NAMES = {0: "Red", 1: "Green", 2: "Blue"}
CHANNEL_EFFECT = {
    "Red": "With R = 0 the image is rendered from Green + Blue only, so it takes on a **cyan / cold cast** — reds become dark or greenish.",
    "Green": "With G = 0 the image is rendered from Red + Blue only, so it takes on a **magenta / purple cast** — greens become dark or brownish.",
    "Blue": "With B = 0 the image is rendered from Red + Green only, so it takes on a **yellow / warm cast** — blues become dark or brownish.",
}


def page_t5() -> None:
    st.markdown('<div class="task-eyebrow">LAB 02 / TASK 05</div>', unsafe_allow_html=True)
    st.title("Color Channel Elimination")
    st.caption("Lab 2 — Pixel & Intensity Operations · Examine the RGB matrix, then zero exactly one channel and observe the result.")

    img, source = image_source("color_cast.png")
    rgb = ops.bgr_to_rgb(img)

    st.markdown("#### Original image")
    show_image(rgb, "Original (RGB)", order="rgb")

    ch_stats = metrics.channel_stats(rgb)
    st.markdown("#### RGB channel statistics")
    st.table(ch_stats)

    means = {r["Channel"]: r["Mean"] for r in ch_stats}
    top_ch = max(means, key=means.get)
    bottom_ch = min(means, key=means.get)
    if means[top_ch] - means[bottom_ch] > 20:
        st.info(
            f"Observed imbalance: the **{top_ch}** channel is the brightest on average "
            f"({means[top_ch]:.1f}) while **{bottom_ch}** is the dimmest ({means[bottom_ch]:.1f}). "
            f"A dominant channel usually signals a **color cast** in that direction; a very "
            f"dim channel may have **lost detail** (e.g. clipped or underexposed content)."
        )
        guidance = (
            f"Guidance for *this* image: the cast is dominated by {top_ch} (highest mean), so "
            f"eliminating **{top_ch}** is the most dramatic way to show how much of the picture's "
            f"appearance that channel carries — the dominant tint is removed from the rendered "
            f"image and the remaining two channels take over. If instead a region looks "
            f"detail-poor, check the dimmest channel ({bottom_ch}): hidden detail *may* live in "
            f"the channel that is currently suppressed. You still choose the final channel below."
        )
    else:
        st.info("Channel means are fairly balanced — no strong cast is present in this image.")
        guidance = (
            f"Guidance for *this* image: the channels are balanced, so any elimination is purely "
            f"diagnostic. If a real image had a cast, pick the **brightest** (dominant) channel "
            f"to remove it; if a real image looked faded/detail-poor, look at the **dimmest** "
            f"channel for hidden information."
        )

    with st.expander("Sample pixels from the RGB matrix (5×5 around center)"):
        matrix_excerpt_table(rgb, size=5)

    ch_name = st.selectbox("Channel to eliminate (set to 0)", list(CHANNEL_NAMES.values()))
    ch_idx = list(CHANNEL_NAMES.values()).index(ch_name)
    out = ops.eliminate_channel(rgb, ch_idx)

    st.markdown("#### Result — one channel zeroed")
    c1, c2 = st.columns(2)
    with c1:
        show_image(rgb, "Original (RGB)", order="rgb")
    with c2:
        st.image(out, caption=f"Result with {ch_name} channel = 0")

    st.markdown(f"**What changed:** {CHANNEL_EFFECT[ch_name]}")
    st.markdown(f"**{guidance}**")
    formula_box(
        "For a pixel (R, G, B):\n"
        f"eliminate Red    -> (0,   G, B)\n"
        f"eliminate Green  -> (R,   0, B)\n"
        f"eliminate Blue   -> (R,   G, 0)\n\n"
        "Exactly one channel is zeroed; the other two are byte-for-byte unchanged.",
        "Rule applied",
    )
    st.markdown(
        "**Important honesty note:** eliminating a channel is a *diagnostic experiment*, not a "
        "quality improvement. The result is deliberately unnatural — it shows *what that channel "
        "was contributing*, not a corrected photo. Real color correction (white balance, "
        "channel scaling) is a different technique."
    )
    discussion([
        "The decision is based on measured channel statistics shown above, not on a preset answer.",
        f"Selected channel: {ch_name}. Effect: the image is re-rendered from the remaining two primaries.",
        "Removing the dominant channel is the strongest demonstration of a color cast; removing the dimmest reveals what that channel was hiding.",
        "Limitation: this is lossy (the removed channel's data is gone in the displayed result) and never a real color correction.",
    ])

    st.markdown("#### Download")
    download_button(out, f"task5_{source_basename(source)}_{ch_name.lower()}_removed.png", order="rgb")


# ---------------------------------------------------------------------------
# Task 6 — histogram enhancement
# ---------------------------------------------------------------------------

def page_t6() -> None:
    st.markdown('<div class="task-eyebrow">LAB 03 / TASK 06</div>', unsafe_allow_html=True)
    st.title("Histogram Enhancement")
    st.caption("Lab 3 — Histogram Enhancement · Start from a low-contrast image, measure its histogram, pick stretching or equalization.")

    img, source = image_source("low_contrast.png")
    if st.checkbox(
        "Demo helper: compress the current image into a narrow intensity band (105–165) to create a low-contrast input",
    ):
        img = ops.make_low_contrast(img, band=(105, 165))
        source += " · compressed"
        st.caption("The working image below is the compressed (low-contrast) version.")

    method = st.radio(
        "Enhancement method",
        ["A — Histogram stretching (linear)", "B — Histogram equalization"],
        horizontal=True,
    )
    delta = st.slider("Optional brightness fine-tuning (applied after enhancement, −50 … +50)", -50, 50, 0)

    if method.startswith("A"):
        out, i_min, i_max = ops.histogram_stretch(img)
        if i_min == i_max:
            st.warning(
                f"The image is **constant** (min = max = {i_min}). Stretching would divide by "
                "zero (range = 0), so it was skipped safely and the image was left unchanged. "
                "Histogram enhancement only makes sense when the image actually varies."
            )
    else:
        out = ops.histogram_equalization(img)
        i_min, i_max = int(luminance(img).min()), int(luminance(img).max())

    out = ops.adjust_brightness(out, delta)

    gray_in = luminance(img)
    gray_out = luminance(out)

    st.markdown("#### Original vs. enhanced")
    c1, c2 = st.columns(2)
    with c1:
        show_image(img, f"Original (intensity range {gray_in.min()}–{gray_in.max()})")
    with c2:
        show_image(out, f"Enhanced — {method} (+ brightness {delta:+d})")

    m1, m2, m3 = st.columns(3)
    m1.metric("Original range", f"{gray_in.min()} – {gray_in.max()}")
    m2.metric("Enhanced range", f"{gray_out.min()} – {gray_out.max()}")
    m3.metric("Contrast (std)", f"{gray_in.std():.1f} → {gray_out.std():.1f}")

    st.pyplot(histogram_figure(
        {"Original (Y luminance)": gray_in, "Enhanced (Y luminance)": gray_out},
        "Histogram before / after — computed from the actual pixel data",
    ))

    stats_table(
        [
            {"Image": "Original", **format_stats(metrics.intensity_stats(gray_in))},
            {"Image": f"Enhanced ({method})", **format_stats(metrics.intensity_stats(gray_out))},
        ],
        "Intensity statistics (luminance)",
    )

    if method.startswith("A"):
        formula_box(
            "Linear stretch using the image's own intensity range:\n\n"
            "out = (I - I_min) * 255 / (I_max - I_min)\n\n"
            f"For this image: I_min = {i_min}, I_max = {i_max}\n"
            "Guard: if I_max == I_min (constant image) the division is skipped.\n\n"
            "For color images the stretch is applied to the Y (luminance) channel of\n"
            "the YCrCb space, so hue (Cr, Cb) is preserved instead of stretching\n"
            "R, G, B independently.",
            "Formula — histogram stretching",
        )
        why = (
            f"**Histogram stretching** was applied. It is a linear remap of the image's actual "
            f"range [{i_min}, {i_max}] onto [0, 255]. Pick it when the problem is a *compressed "
            f"range* (sensor underexposure, flat foggy light, a narrow histogram) and you want a "
            f"predictable, order-preserving fix: bright stays brighter, relative tonal spacing is "
            f"kept, and no histogram shape assumptions are made."
        )
    else:
        formula_box(
            "Histogram equalization redistributes intensities so the output histogram\n"
            "is as flat (uniform) as possible, via the CDF of the input histogram:\n\n"
            "out(v) = round( 255 * (CDF(v) - CDF_min) / (N - CDF_min) )\n\n"
            "Implemented here with cv2.equalizeHist on the Y (luminance) channel of\n"
            "YCrCb; the chrominance channels are kept, which avoids the color\n"
            "distortion of equalizing R, G, B separately.",
            "Concept — histogram equalization",
        )
        why = (
            "**Histogram equalization** was applied. It redistributes the pixel values using the "
            "histogram's cumulative distribution, so whatever narrow band the image occupied gets "
            "spread across the full 0–255 scale. Pick it when the histogram is *concentrated in a "
            "narrow region* (the classic low-contrast photo) — it adapts to the data instead of "
            "using the raw min/max, and often recovers more visible detail in the main content. "
            "Trade-off: it can over-emphasize noise and push small background regions to the "
            "extremes."
        )

    st.markdown(
        f"""
**Contrast vs. brightness.** *Brightness* is the overall level of the intensities
(the histogram's position); *contrast* is how spread out they are (the histogram's
width). Stretching and equalization change **contrast**; the optional slider above
changes **brightness** — the two effects are independent, which is why both are offered.

{why}
"""
    )
    discussion([
        "Stretching: linear, depends only on the image's min/max — simple and predictable, but one outlier pixel can dominate the scale.",
        "Equalization: adaptive (uses the full CDF) — stronger on concentrated histograms, but amplifies noise and can exaggerate background.",
        "Color handling: work on Y in YCrCb, never equalize R/G/B independently (hue would drift).",
        f"For this image (range {gray_in.min()}–{gray_in.max()}, std {gray_in.std():.1f}) the method shown above was selected; the statistics and histograms on this page are the evidence to cite.",
    ])

    st.markdown("#### Download")
    tag = "stretch" if method.startswith("A") else "equalize"
    download_button(out, f"task6_{source_basename(source)}_{tag}.png")


# ---------------------------------------------------------------------------
# Task 7 — noise reduction + SNR
# ---------------------------------------------------------------------------

def format_snr(v: float) -> str:
    return "∞ (identical to reference)" if v == float("inf") else f"{v:.2f} dB"


def page_t7() -> None:
    st.markdown('<div class="task-eyebrow">LAB 04–05 / TASK 07</div>', unsafe_allow_html=True)
    st.title("Noise Reduction & SNR")
    st.caption("Labs 4–5 — Filtering · Inject artificial noise into a clean image, remove it with a suitable filter, measure SNR from scratch.")

    clean, source = image_source("detail_scene.png")

    st.markdown(
        '<div class="source-heading">EXPERIMENT SETTINGS</div>',
        unsafe_allow_html=True,
    )
    noise_col, strength_col, filter_col = st.columns([1.2, 1, 1.1], gap="large")
    with noise_col:
        noise_type = st.radio(
            "Noise type",
            ["Salt-and-pepper (impulse) noise", "Gaussian (additive) noise"],
        )
        seed = st.number_input(
            "Random seed", 0, 1_000_000, 42, step=1,
            help="The same seed produces the same synthetic noise pattern.",
        )

    with strength_col:
        if noise_type.startswith("Salt"):
            ratio = st.slider("Corrupted pixel ratio", 0.01, 0.50, 0.08, 0.01)
            noisy = ops.add_salt_and_pepper(clean, ratio, int(seed))
            recommended = "Median filter"
        else:
            sigma = st.slider("Gaussian noise σ", 5, 60, 15, 1)
            noisy = ops.add_gaussian_noise(clean, sigma, int(seed))
            recommended = "Gaussian filter"
        st.caption(f"Recommended: {recommended}")

    with filter_col:
        filter_name = st.selectbox(
            "Smoothing filter",
            list(ops.SMOOTHING_FILTERS.keys()),
            index=list(ops.SMOOTHING_FILTERS.keys()).index(recommended),
        )
        ksize = st.slider("Filter kernel size (odd)", 3, 15, 5, 2)
    denoised = ops.SMOOTHING_FILTERS[filter_name](noisy, iu.odd_kernel(ksize))

    if filter_name != recommended:
        st.caption(f"Note: the filter recommended for {noise_type.split(' (')[0].lower()} noise is the **{recommended}** — you have chosen a different one, which is fine for comparison.")

    st.markdown("#### Clean · Noisy · Denoised")
    c1, c2, c3 = st.columns(3)
    with c1:
        show_image(clean, "Clean (reference)")
    with c2:
        show_image(noisy, f"Noisy — {noise_type}")
    with c3:
        show_image(denoised, f"Denoised — {filter_name}, k={iu.odd_kernel(ksize)}")

    parts_noisy = metrics.snr_parts(clean, noisy)
    parts_den = metrics.snr_parts(clean, denoised)
    m1, m2 = st.columns(2)
    m1.metric("SNR — before denoising (noisy vs clean)", format_snr(parts_noisy["snr_db"]))
    snr_before = parts_noisy["snr_db"]
    snr_after = parts_den["snr_db"]
    improvement = (
        f"{snr_after - snr_before:+.2f} dB"
        if np.isfinite(snr_before) and np.isfinite(snr_after)
        else None
    )
    m2.metric(
        "SNR — after denoising (denoised vs clean)",
        format_snr(snr_after),
        delta=improvement,
        delta_color="normal",
    )

    with st.expander("Inspect the SNR calculation values (same clean reference)"):
        st.table(
            [
                {"Quantity": "Signal power  Σ S²", "Noisy": f"{parts_noisy['signal_power']:.0f}", "Denoised": f"{parts_den['signal_power']:.0f}"},
                {"Quantity": "Noise power  Σ (S−X)²", "Noisy": f"{parts_noisy['noise_power']:.0f}", "Denoised": f"{parts_den['noise_power']:.0f}"},
                {"Quantity": "SNR (dB)", "Noisy": format_snr(parts_noisy["snr_db"]), "Denoised": format_snr(parts_den["snr_db"])},
            ]
        )

    why_filter = (
        f"The **{filter_name}** was used. The recommendation logic: *salt-and-pepper* noise "
        "replaces individual pixels by extreme values (0 or 255) — the **median** of a small "
        "window discards such outliers, which is why it is the textbook choice; *Gaussian* "
        "noise is additive and spread over all pixels — a **Gaussian** (or mean) filter matches "
        "its statistics and averages it out, while a median would only partially help. "
        f"Here you manually selected **{filter_name}**, and the two SNR values above are the "
        "objective evidence of whether that selection worked for this image."
    )
    formula_box(
        "SNR from scratch (implemented in src/metrics.py, NOT a library function):\n\n"
        "  signal power  = Σ S²            S = clean reference image\n"
        "  noise power   = Σ (S - X)²      X = evaluated image (noisy or denoised)\n"
        "  SNR (dB)      = 10 · log10( signal power / noise power )\n\n"
        "Computed in float64. If X == S exactly, noise power = 0 -> SNR = ∞ (perfect).\n"
        "This is a power-ratio SNR — deliberately NOT PSNR (PSNR would divide by\n"
        "the MSE and use 255² as the reference level). Higher SNR = less error\n"
        "relative to the reference signal.",
        "SNR calculation",
    )
    st.markdown(f"**Why this filter:** {why_filter}")
    st.markdown(
        "**Interpretation:** the *before* SNR quantifies the damage the noise did; the *after* "
        "SNR quantifies how much of the clean signal the filter recovered. A good denoiser "
        "raises the SNR (reduces noise power). Note that any smoothing also removes real "
        "high-frequency detail, so the denoised image is a trade-off: less noise, slightly "
        "softer edges."
    )
    discussion([
        f"Noise model: {noise_type}, generated with NumPy and seed {int(seed)} — the same seed always reproduces the same noisy image.",
        f"Filter: {filter_name} (kernel {iu.odd_kernel(ksize)}×{iu.odd_kernel(ksize)}). Recommended for this noise type: {recommended}.",
        "SNR uses the identical clean reference for both measurements, so the two numbers are directly comparable.",
        "A median filter is robust to impulses because the middle value ignores outliers; Gaussian smoothing models additive noise; mean is a cheap but blurrier baseline.",
        "Limitation: smoothing reduces noise power but also removes genuine detail — SNR going up does not mean the image is 'perfect'.",
    ])

    st.markdown("#### Download")
    c1, c2 = st.columns(2)
    with c1:
        download_button(noisy, f"task7_{source_basename(source)}_noisy.png", label="Download Noisy Image")
    with c2:
        download_button(denoised, f"task7_{source_basename(source)}_denoised_{filter_name.split()[0].lower()}.png", label="Download Denoised Image")


# ---------------------------------------------------------------------------
# Task 8 — blur & sharpening
# ---------------------------------------------------------------------------

KERNEL_LAPLACIAN_STR = (
    "[ 0  -1   0 ]\n"
    "[ -1   4  -1 ]   (4-neighbour Laplacian, +4 centre)\n"
    "[ 0  -1   0 ]\n\n"
    "response = I ⊗ L   (computed in float32, signed, no truncation)\n"
    "out = I + α · response      # positive where a pixel is brighter than"
    "                             # its neighbours -> bright features get brighter"
)
KERNEL_CUSTOM_STR = (
    "base kernel  K_base = [ 0  -1   0 ]\n"
    "                     [ -1   5  -1 ]\n"
    "                     [ 0  -1   0 ]\n\n"
    "effective kernel:  K(s) = δ + s · (K_base − δ)\n"
    "(δ = unit impulse, the identity element of convolution; sum δ = 1)\n"
    "sum(K(s)) = 1 for every s  ->  constant regions stay constant at any\n"
    "strength; s = 0 leaves the image unchanged, only the high-pass part\n"
    "is scaled with s.\n"
    "out = blurred ⊗ K(s)   (float32, clipped to 0–255)"
)


def page_t8() -> None:
    st.markdown('<div class="task-eyebrow">LAB 04–05 / TASK 08</div>', unsafe_allow_html=True)
    st.title("Detail Enhancement (Blur → Sharpening)")
    st.caption("Labs 4–5 — Filtering · Deliberately blur a clean image, then sharpen it and see what comes back.")

    clean, source = image_source("detail_scene.png")

    c1, c2 = st.columns(2)
    with c1:
        blur_k = st.slider("Gaussian blur kernel size (odd)", 3, 21, 7, 2)
    with c2:
        method = st.radio(
            "Sharpening technique",
            ["Laplacian-based sharpening", "Custom 3×3 sharpening kernel", "Unsharp masking"],
            horizontal=False,
        )

    blurred = ops.gaussian_blur(clean, iu.odd_kernel(blur_k))

    if method.startswith("Laplacian"):
        alpha = st.slider("Sharpening strength α (0 – 2)", 0.0, 2.0, 1.0, 0.1)
        sharpened = ops.sharpen_laplacian(blurred, alpha)
        kernel_block = KERNEL_LAPLACIAN_STR + f"\n\nα = {alpha}"
        explainer = (
            f"**Laplacian sharpening** with α = {alpha}: the +4-centre 4-neighbour Laplacian is "
            "a second-derivative (band-pass) operator whose response is *positive* where a pixel "
            "is brighter than its surroundings. The convolution is computed in float32 (signed, "
            "so nothing is truncated), and α times that response is **added** to the image — "
            "bright features get even brighter, so edges and fine detail are emphasized. "
            "α = 0 returns the input unchanged; larger α increases edge emphasis (and ringing) "
            "at the cost of stability."
        )
    elif method.startswith("Custom"):
        intensity = st.slider("Kernel strength s (0.5 – 2)", 0.5, 2.0, 1.0, 0.1)
        sharpened = ops.sharpen_custom_kernel(blurred, intensity)
        kernel_block = KERNEL_CUSTOM_STR + f"\n\ns = {intensity}\n"
        explainer = (
            f"**Custom 3×3 kernel sharpening** with strength {intensity}: the base kernel weights "
            "the centre pixel +5 and subtracts the four neighbours (−1 each). The effective "
            "kernel K(s) = δ + s·(K_base − δ) (δ = unit impulse) keeps its sum equal to 1 at "
            "*any* strength, so flat regions are exactly unchanged (no DC gain shift) while only "
            "the high-pass part — the local contrast around edges — grows with s."
        )
    else:
        amount = st.slider("Unsharp amount k (0.2 – 2)", 0.2, 2.0, 1.0, 0.1)
        sharpened = ops.unsharp_mask(blurred, amount, iu.odd_kernel(blur_k))
        kernel_block = ("out = I + k · (I - blurred(I))\n\n"
                        f"high-pass detail = I - blurred(I)   (Gaussian blur, kernel {iu.odd_kernel(blur_k)})\n"
                        f"k = {amount}")
        explainer = (
            f"**Unsharp masking** with k = {amount}: the difference between the image and its "
            "blurred copy is a high-pass (detail) signal; adding a scaled copy back to the "
            "image increases local contrast along edges. It is the same idea as the other two "
            "methods, implemented through subtraction instead of a fixed kernel."
        )

    st.markdown("#### Original (clean) · Blurred · Sharpened")
    c1, c2, c3 = st.columns(3)
    with c1:
        show_image(clean, "Original clean image")
    with c2:
        show_image(blurred, f"Deliberately blurred (Gaussian, k={iu.odd_kernel(blur_k)})")
    with c3:
        show_image(sharpened, "Sharpened")

    formula_box(kernel_block, "Kernel / expression used")

    st.markdown(f"**Why this technique was chosen:** {explainer}")
    st.markdown(
        f"The image was first **blurred on purpose** (Gaussian, kernel {iu.odd_kernel(blur_k)}) so "
        "that the sharpening has a visible job to do. **Scientific honesty:** Gaussian blur is a "
        "low-pass filter — it permanently discards high-frequency information (information is "
        "lost, not just hidden). Sharpening re-emphasizes the high frequencies that *survived* "
        "the blur, so the result can look crisper, but it **cannot reconstruct** the detail the "
        "blur destroyed, and it will also amplify any noise present. That is why real pipelines "
        "sharpen carefully and never treat sharpening as an undo button."
    )
    discussion([
        f"Blur first: Gaussian kernel {iu.odd_kernel(blur_k)}×{iu.odd_kernel(blur_k)} — a controlled, reproducible loss of high-frequency detail.",
        f"Sharpening method: {method}. Kernel/expression shown in the formula box; parameters were chosen interactively.",
        "Blur = information loss (low-pass). Sharpening = high-frequency emphasis. The two do not exactly cancel each other.",
        "Limitation: over-sharpening produces halos/ringing and amplifies noise; the 'improvement' is perceptual, not a true reconstruction.",
    ])

    st.markdown("#### Download")
    c1, c2 = st.columns(2)
    with c1:
        download_button(blurred, f"task8_{source_basename(source)}_blurred.png")
    with c2:
        download_button(sharpened, f"task8_{source_basename(source)}_sharpened_{method.split()[0].lower()}.png")


# ---------------------------------------------------------------------------
# Task 9 — edge detection
# ---------------------------------------------------------------------------

SOBEL_KX_STR = (
    "Sobel X (horizontal gradient)     Sobel Y (vertical gradient)\n\n"
    "  [ -1   0   1 ]                 [ -1  -2  -1 ]\n"
    "  [ -2   0   2 ]                 [  0   0   0 ]\n"
    "  [ -1   0   1 ]                 [  1   2   1 ]\n\n"
    "  gx = I ⊗ Kx      gy = I ⊗ Ky      magnitude = sqrt(gx² + gy²)\n\n"
    "(The standard 3×3 Sobel kernels as implemented by cv2.Sobel. OpenCV's\n"
    " row-first coordinate convention may flip the sign of gx compared with\n"
    " the textbook ∂I/∂x — the magnitude, which is what is displayed, is\n"
    " unaffected by that sign.)"
)
CANNY_STEPS_STR = ("Canny — 5 stages:\n"
                   "1) Gaussian smoothing (remove noise)\n"
                   "2) gradient magnitude + direction (like Sobel)\n"
                   "3) non-maximum suppression (thin edges to 1 px)\n"
                   "4) double thresholding (low / high)\n"
                   "5) hysteresis: keep weak edges only if connected to strong edges")
LAPLACIAN_EDGE_STR = ("Laplacian edge operator (second derivative, isotropic):\n\n"
                      "  [  0  -1   0 ]\n"
                      "  [ -1   4  -1 ]\n"
                      "  [  0  -1   0 ]\n\n"
                      "Response shown as |Laplacian(I)| — both rising and falling edges respond.")


def page_t9() -> None:
    st.markdown('<div class="task-eyebrow">LAB 04–05 / TASK 09</div>', unsafe_allow_html=True)
    st.title("Edge Detection")
    st.caption("Labs 4–5 — Filtering · Choose an operator that reveals the structural boundaries of the image.")

    img, source = image_source("geometric_edges.png")
    gray = iu.to_gray(img)

    method = st.radio("Edge operator", ["Sobel", "Canny", "Laplacian"], horizontal=True)

    params = {}
    if method == "Sobel":
        params["ksize"] = st.slider("Sobel kernel size (odd)", 3, 7, 3, 2)
        edges = ops.sobel_edges(gray, params["ksize"])
        explainer = (
            f"**Sobel** (kernel {params['ksize']}×{params['ksize']}) approximates the first "
            "derivative in x and y; the displayed map is the normalized gradient "
            "*magnitude* √(gx²+gy²) — using the magnitude (instead of the signed gradients) "
            "avoids incorrectly truncating negative values. Sobel responds to the *strength* "
            "of intensity change, so edges appear as bright, slightly thick bands, and the "
            "two gradient directions reveal whether a boundary is horizontal, vertical or "
            "diagonal."
        )
        formula_block = SOBEL_KX_STR
        when = ("Prefer Sobel when you need gradient strength/direction (texture analysis, "
                "orientation of boundaries, a quick structural map).")
    elif method == "Canny":
        c1, c2 = st.columns(2)
        params["low"] = c1.slider("Low threshold", 0, 200, 50, 5)
        params["high"] = c2.slider("High threshold", 10, 255, 150, 5)
        edges = ops.canny_edges(gray, params["low"], params["high"])
        explainer = (
            f"**Canny** (low = {params['low']}, high = {params['high']}) is a multi-stage "
            "detector: smooth, compute gradients, thin them with non-maximum suppression, then "
            "keep only edges that pass the double-threshold hysteresis test. The result is a "
            "binary map of *thin, continuous, well-located* boundaries — the strongest edges "
            "are kept automatically and weak ones only when they connect to strong ones. Raise "
            "the high threshold to keep only the strongest boundaries; lower it to reveal "
            "fainter structure (at the cost of more false edges)."
        )
        formula_block = CANNY_STEPS_STR + f"\n\nlow = {params['low']}, high = {params['high']}"
        when = ("Prefer Canny when you need clean, 1-pixel-thick boundaries for segmentation "
                "or contour extraction — it suppresses weak noise better than any single kernel.")
    else:
        edges = ops.laplacian_edges(gray)
        explainer = (
            "**Laplacian** is the second derivative (isotropic — it responds to edges in any "
            "direction at once). Because second derivatives cross zero at an edge, the response "
            "appears as *double* bright bands around each boundary, and the operator is more "
            "sensitive to noise (which is why it is usually applied after smoothing)."
        )
        formula_block = LAPLACIAN_EDGE_STR
        when = ("Use the Laplacian for a quick, direction-agnostic edge response or as a "
                "building block (e.g. Laplacian of Gaussian for blob detection).")

    st.markdown("#### Original vs. edge map")
    c1, c2 = st.columns(2)
    with c1:
        st.image(gray, channels="grayscale", caption="Grayscale original")
    with c2:
        st.image(edges, channels="grayscale", caption=f"Edge map — {method}")

    formula_box(formula_block, f"Operator — {method}")
    st.markdown(f"**Why this operator was chosen:** {explainer}")
    st.markdown(f"**When to prefer it:** {when} Structural note: on this image you can check that the operator traces exactly the square, the circle, the diagonal, the text and the small dot — that is what 'revealing structural boundaries' means, and different operators do it with different thickness and cleanliness.")

    discussion([
        f"Operator: {method} with the parameters shown. The edge map is generated live from the grayscale conversion of the displayed image.",
        "Sobel = first derivative (gradient strength, directional); Canny = 5-stage detector with hysteresis (thin, robust); Laplacian = second derivative (isotropic, double edges, noise-sensitive).",
        "Sobel is displayed as a normalized magnitude (0–255) so negative gradients are never truncated incorrectly.",
        "Limitation: all edge detectors respond to *intensity* changes — texture, noise and gradual shading can produce false edges; parameter choice (thresholds, kernel size) always trades completeness against precision.",
    ])

    st.markdown("#### Download")
    download_button(edges, f"task9_{source_basename(source)}_{method.lower()}_edges.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    pages = {
        "dashboard": page_dashboard,
        "t1": page_t1,
        "t2": page_t2,
        "t3": page_t3,
        "t4": page_t4,
        "t5": page_t5,
        "t6": page_t6,
        "t7": page_t7,
        "t8": page_t8,
        "t9": page_t9,
    }
    nav_key = render_sidebar()
    pages[nav_key]()


main()
