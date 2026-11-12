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
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .lab-card { border: 1px solid #d7e0ea; border-radius: 12px;
                  padding: 14px 16px; background: #ffffff; }
      .discuss { background: #f0f6ff; border-left: 4px solid #2563eb;
                 padding: 10px 14px; border-radius: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)

NAV: list[tuple[str, str]] = [
    ("dashboard", "🏠 Dashboard"),
    ("t1", "Lab 1 · Task 1 — RGB Channel Separation"),
    ("t2", "Lab 1 · Task 2 — Custom Grayscale"),
    ("t3", "Lab 2 · Task 3 — Arithmetic Operations"),
    ("t4", "Lab 2 · Task 4 — Negative / Solarization"),
    ("t5", "Lab 2 · Task 5 — Channel Elimination"),
    ("t6", "Lab 3 · Task 6 — Histogram Enhancement"),
    ("t7", "Labs 4–5 · Task 7 — Noise Reduction + SNR"),
    ("t8", "Labs 4–5 · Task 8 — Blur & Sharpening"),
    ("t9", "Labs 4–5 · Task 9 — Edge Detection"),
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
                format_func=lambda s: f"{s}  ·  {iu.SAMPLE_INFO.get(s, '')}",
                label_visibility="collapsed",
            )
    with c2:
        up = st.file_uploader(
            "Or upload your own image",
            type=ACCEPTED_FORMATS,
            label_visibility="collapsed",
        )

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


def show_image(img: np.ndarray, caption: str | None = None) -> None:
    """Display a BGR color or 2-D gray image with the right channel order."""
    if img.ndim == 2:
        st.image(img, channels="grayscale", caption=caption)
    else:
        st.image(ops.bgr_to_rgb(img), caption=caption)


def download_button(img: np.ndarray, name: str) -> None:
    st.download_button(
        "⬇️  Download PNG",
        iu.encode_png(img),
        file_name=name,
        mime="image/png",
        use_container_width=True,
    )


def formula_box(text: str, title: str = "Formula / mathematical rule") -> None:
    with st.expander(f"📐 {title}"):
        st.code(text, language=None)


def discussion(points: list[str], title: str = "Discussion notes (for oral evaluation)") -> None:
    with st.expander(f"🎓 {title}", expanded=False):
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


def matrix_excerpt_table(rgb: np.ndarray, size: int = 5) -> None:
    """Show a small (size x size) RGB matrix excerpt around the image center."""
    h, w = rgb.shape[:2]
    y0 = max(0, h // 2 - size // 2)
    x0 = max(0, w // 2 - size // 2)
    rows = []
    for y in range(y0, min(h, y0 + size)):
        for x in range(x0, min(w, x0 + size)):
            r, g, b = (int(v) for v in rgb[y, x])
            rows.append({"Pixel (row, col)": f"({y}, {x})", "R": r, "G": g, "B": b})
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
    with st.sidebar:
        st.markdown("## 🧪 Image Processing Lab Studio")
        st.caption("University image-processing course · 9 tasks · 4 labs")
        st.divider()
        st.markdown(
            """
**Labs**
- **Lab 1** — Image fundamentals (Tasks 1–2)
- **Lab 2** — Pixel & intensity operations (Tasks 3–5)
- **Lab 3** — Histogram enhancement (Task 6)
- **Labs 4–5** — Filtering & edge detection (Tasks 7–9)
            """
        )
        labels = [label for _, label in NAV]
        st.radio("Navigate", labels, index=0, key="nav", label_visibility="collapsed")
        st.divider()
        st.caption(
            "Demonstration images are bundled in `assets/samples/` — "
            "the app never needs internet access. You can upload your own "
            "image on every task page."
        )
    # In a real Streamlit run the radio widget always fills session state;
    # the fallback keeps a bare `python app.py` import from crashing.
    nav = st.session_state.get("nav")
    if nav not in NAV_BY_LABEL:
        nav = NAV[0][1]
    return NAV_BY_LABEL[nav]


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

def page_dashboard() -> None:
    st.title("🧪 Image Processing Lab Studio")
    st.markdown(
        "An interactive, educational workbench that implements **nine university "
        "image-processing tasks** in one Streamlit application. Pick a task, load a "
        "bundled demonstration image (or upload your own), adjust parameters, compare "
        "original and processed results, and read the mathematics and the reasoning "
        "behind every choice — everything you need for a practical evaluation and an "
        "oral discussion."
    )

    st.markdown("### Lab overview")
    cards = [
        ("🎞️ Lab 1 — Image Fundamentals", [
            "**Task 1** · RGB Channel Separation",
            "**Task 2** · Custom Weighted Grayscale",
        ]),
        ("➗ Lab 2 — Pixel & Intensity Operations", [
            "**Task 3** · Arithmetic Enhancement",
            "**Task 4** · Negative / Solarization",
            "**Task 5** · Color Channel Elimination",
        ]),
        ("📊 Lab 3 — Histogram Enhancement", [
            "**Task 6** · Histogram Stretching & Equalization",
        ]),
        ("🔍 Labs 4 & 5 — Filtering & Edge Detection", [
            "**Task 7** · Noise Reduction + SNR (from scratch)",
            "**Task 8** · Blur & Sharpening",
            "**Task 9** · Edge Detection (Sobel / Canny / Laplacian)",
        ]),
    ]
    cols = st.columns(2)
    for i, (title, items) in enumerate(cards):
        with cols[i % 2]:
            st.markdown(f'<div class="lab-card"><b>{title}</b><br>{"<br>".join(items)}</div>',
                        unsafe_allow_html=True)

    st.markdown("### All nine tasks")
    done = [
        "Task 1 — RGB Channel Separation (BGR↔RGB, channels, isolated colors, pixel inspector, matrix excerpt)",
        "Task 2 — Custom Grayscale via `Gray = 0.299R + 0.587G + 0.114B` (manual NumPy, float, clip)",
        "Task 3 — Arithmetic operations `I+C`, `I−C`, `I×C`, `I÷C` with overflow-safe clipping",
        "Task 4 — Negative `255−I` and threshold-controlled Solarization",
        "Task 5 — Elimination of exactly one RGB channel with statistics evidence",
        "Task 6 — Histogram Stretching (linear) and Equalization (Y-channel for color), live histograms",
        "Task 7 — Salt & pepper / Gaussian noise + Median / Gaussian / Mean filters, manual SNR in dB",
        "Task 8 — Gaussian blur then Laplacian / custom-kernel / unsharp-mask sharpening",
        "Task 9 — Sobel, Canny and Laplacian edge detection with normalized visualization",
    ]
    for d in done:
        st.markdown(f"✅ {d}")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Typical workflow")
        st.markdown(
            """1. **Choose a task** in the sidebar (or use quick navigation below).
2. **Load an image** — bundled demonstration image or your own upload.
3. **Tune the parameters** with sliders and selectors.
4. **Compare** original vs. processed images.
5. **Inspect the numbers** — statistics, histograms, SNR, matrix excerpts.
6. **Read the formula** and the *why this technique* explanation.
7. **Review the discussion notes** prepared for oral evaluation.
8. **Download** the processed image as PNG."""
        )
    with c2:
        st.markdown("### Technology stack")
        st.markdown(
            """- Python 3.10+
- Streamlit (interactive UI)
- OpenCV (`opencv-python-headless`)
- NumPy (all pixel math in float, then clipped to uint8)
- Matplotlib (histograms and plots)
- Pillow (independent upload validation)
- Pytest (automated tests)

> Educational tool: deterministic classical image processing.
> No deep learning, no external APIs, no internet access at runtime."""
        )
        st.markdown("### Run locally")
        st.code(
            "python -m venv .venv\npip install -r requirements.txt\nstreamlit run app.py",
            language="bash",
        )

    st.divider()
    jump = st.selectbox(
        "⚡ Quick navigation — jump straight to a task",
        ["— choose a task —"] + [label for _, label in NAV[1:]],
    )
    if jump != "— choose a task —":
        key = NAV_BY_LABEL[jump]
        if st.session_state.get("nav") != key:
            st.session_state["nav"] = key
            st.rerun()


# ---------------------------------------------------------------------------
# Task 1 — RGB channel separation
# ---------------------------------------------------------------------------

def page_t1() -> None:
    st.title("Task 1 · RGB Channel Separation")
    st.caption("Lab 1 — Image Fundamentals · Extract and inspect the Red, Green and Blue channels of a color image.")

    img, source = image_source("color_scene.png")
    rgb = ops.bgr_to_rgb(img)
    r, g, b = ops.separate_channels(rgb)

    st.markdown("#### Channel order: BGR in OpenCV vs. RGB for display")
    st.markdown(
        "OpenCV loads color images in **BGR** order. If a BGR array is shown as if it were RGB, "
        "the **red and blue channels appear swapped**. The left image below is the raw OpenCV "
        "array mis-displayed; the right image is the correctly converted RGB version."
    )
    c1, c2 = st.columns(2)
    with c1:
        # Show the BGR array *as if* it were RGB: the classic red/blue swap demo.
        st.image(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                 caption="Raw OpenCV BGR array displayed without conversion (red/blue look swapped)")
    with c2:
        st.image(rgb, caption="Correctly converted: `cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)`")

    st.markdown("#### 1 · Original image (RGB)")
    show_image(ops.bgr_to_rgb(rgb), "Original")

    tab_map, tab_iso, tab_px, tab_mx = st.tabs(
        ["2 · Channel intensity maps", "3 · Color-isolated images", "4 · Pixel inspector", "5 · RGB matrix excerpt"]
    )
    with tab_map:
        st.markdown("Each channel shown as a grayscale **intensity map**: bright = that primary is strong there.")
        ch = st.columns(3)
        for col, arr, name in zip(ch, (r, g, b), ("Red", "Green", "Blue")):
            with col:
                st.image(arr, channels="grayscale", caption=f"{name} channel — intensity only")
    with tab_iso:
        st.markdown("Color-isolated images: only the corresponding channel is preserved, the other two are set to 0.")
        ch = st.columns(3)
        for col, i, name in zip(ch, (0, 1, 2), ("Only Red", "Only Green", "Only Blue")):
            with col:
                iso = ops.isolate_channel(rgb, i)
                st.image(iso, caption=name)
    with tab_px:
        st.markdown("Inspect the exact stored values of any pixel.")
        h, w = rgb.shape[:2]
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            px = st.slider("Column x", 0, max(0, w - 1), min(50, max(0, w - 1)))
        with c2:
            py = st.slider("Row y", 0, max(0, h - 1), min(50, max(0, h - 1)))
        r_v, g_v, b_v = (int(v) for v in rgb[py, px])
        with c3:
            m1, m2, m3 = st.columns(3)
            m1.metric("R", r_v)
            m2.metric("G", g_v)
            m3.metric("B", b_v)
            marked = rgb.copy()
            cv2.circle(marked, (px, py), 4, (255, 0, 0), 2)
            st.image(marked, caption=f"Selected pixel at (row {py}, col {px})")
    with tab_mx:
        st.markdown("The image *is* a matrix of (R, G, B) triplets. A 5×5 excerpt around the center:")
        matrix_excerpt_table(rgb, size=5)

    formula_box(
        "Pixel = (R, G, B) with R, G, B in [0, 255]\n"
        "OpenCV stores (B, G, R)  ->  rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)\n"
        "r, g, b = cv2.split(rgb)            # three single-channel arrays\n"
        "isolate(i) = merge(ch0 if i==0 else 0, ch1 if i==1 else 0, ch2 if i==2 else 0)",
        "Formula / operations used",
    )

    st.markdown(
        f"""
**What is the RGB representation?** Every color pixel is stored as three intensity
values — Red, Green and Blue — each 0 (off) to 255 (full). The perceived color is the
mix of the three; for example `{r_v}, {g_v}, {b_v}` at the selected pixel. Separating the
channels shows each primary's *contribution* to the scene: the red channel is bright where
the image contains red, and dark where it does not, and so on for green and blue.
**Why this technique?** Channel separation is the most basic way to inspect what an image
is made of; it is also the prerequisite for every channel-based operation in the other tasks.
"""
    )
    discussion([
        "OpenCV uses BGR; always convert before display or analysis that assumes RGB order.",
        "A channel intensity map is grayscale: bright = that primary is strong in that region.",
        "Color isolation (keeping one channel, zeroing the others) shows what the scene looks like under that single primary color.",
        "Limitation: the channel split is exact and lossless — no information is created or destroyed; it only re-presents the same data.",
    ])

    st.markdown("#### Download")
    base = source_basename(source)
    c1, c2, c3 = st.columns(3)
    with c1:
        download_button(ops.isolate_channel(rgb, 0), f"task1_{base}_red_only.png")
    with c2:
        download_button(ops.isolate_channel(rgb, 1), f"task1_{base}_green_only.png")
    with c3:
        download_button(ops.isolate_channel(rgb, 2), f"task1_{base}_blue_only.png")


# ---------------------------------------------------------------------------
# Task 2 — custom weighted grayscale
# ---------------------------------------------------------------------------

def page_t2() -> None:
    st.title("Task 2 · Custom Grayscale Conversion")
    st.caption("Lab 1 — Image Fundamentals · Grayscale via the weighted luminance formula, implemented manually (no average, no cv2 conversion).")

    img, source = image_source("color_scene.png")
    rgb = ops.bgr_to_rgb(img)
    gray_custom = ops.weighted_grayscale(rgb)
    gray_cv = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)  # reference for comparison only
    max_diff = int(np.max(np.abs(gray_custom.astype(np.int16) - gray_cv.astype(np.int16))))

    st.markdown("#### Original vs. custom weighted grayscale")
    c1, c2, c3 = st.columns(3)
    with c1:
        show_image(rgb, "Original (RGB)")
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
    st.title("Task 3 · Arithmetic Image Enhancement")
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
               "histogram to the right and recovers shadow detail at a constant offset, which "
               "keeps the tonal *spacing* unchanged.")
    elif op == "sub":
        effect = (f"Every pixel was lowered by {c_val}. Overexposed highlights are pulled back; "
                  f"pixels below {c_val} clip to black." if c_val > 0 else
                  f"Every pixel was lifted by {abs(c_val)} (C is negative).")
        why = ("Subtraction is the classic choice for an **overexposed** image: it shifts the "
               "histogram to the left, recovering highlight detail at a constant offset.")
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
    st.title("Task 4 · Complement (Negative) & Solarization")
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
        frac = float(np.mean(luminance(img) >= threshold)) * 100
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
            f"**Solarization with T = {threshold}** was selected. Because only pixels with "
            f"I ≥ T are inverted, about **{frac:.1f}% of the luminance pixels** of this image "
            f"are affected — the highlights flip while the shadows stay put. That makes it the "
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
    st.title("Task 5 · Color Channel Elimination")
    st.caption("Lab 2 — Pixel & Intensity Operations · Examine the RGB matrix, then zero exactly one channel and observe the result.")

    img, source = image_source("color_cast.png")
    rgb = ops.bgr_to_rgb(img)

    st.markdown("#### Original image")
    show_image(rgb, "Original")

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
            f"appearance that channel carries — the cast disappears and the remaining two "
            f"channels take over. If instead a region looks "
            f"detail-poor, check the dimmest channel ({bottom_ch}): hidden detail often lives in "
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

    with st.expander("🔢 Sample pixels from the RGB matrix (5×5 around center)"):
        matrix_excerpt_table(rgb, size=5)

    ch_name = st.selectbox("Channel to eliminate (set to 0)", list(CHANNEL_NAMES.values()))
    ch_idx = list(CHANNEL_NAMES.values()).index(ch_name)
    out = ops.eliminate_channel(rgb, ch_idx)

    st.markdown("#### Result — one channel zeroed")
    c1, c2 = st.columns(2)
    with c1:
        show_image(rgb, "Original")
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
    download_button(out, f"task5_{source_basename(source)}_{ch_name.lower()}_removed.png")


# ---------------------------------------------------------------------------
# Task 6 — histogram enhancement
# ---------------------------------------------------------------------------

def page_t6() -> None:
    st.title("Task 6 · Histogram Enhancement")
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
    st.title("Task 7 · Noise Reduction & SNR")
    st.caption("Labs 4–5 — Filtering · Inject artificial noise into a clean image, remove it with a suitable filter, measure SNR from scratch.")

    clean, source = image_source("detail_scene.png")

    c1, c2 = st.columns(2)
    with c1:
        seed = st.number_input("Random seed (reproducible experiments)", 0, 1_000_000, 42, step=1)
        noise_type = st.radio(
            "Noise type",
            ["Salt-and-pepper (impulse) noise", "Gaussian (additive) noise"],
            horizontal=True,
        )
    with c2:
        if noise_type.startswith("Salt"):
            ratio = st.slider("Corrupted-pixel ratio (0.01 – 0.50)", 0.01, 0.50, 0.08, 0.01)
            noisy = ops.add_salt_and_pepper(clean, ratio, int(seed))
            recommended = "Median filter"
        else:
            sigma = st.slider("Gaussian standard deviation σ (5 – 60)", 5, 60, 15, 1)
            noisy = ops.add_gaussian_noise(clean, sigma, int(seed))
            recommended = "Gaussian filter"

    filter_name = st.selectbox(
        "Smoothing filter (manual selection allowed)",
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
    m2.metric("SNR — after denoising (denoised vs clean)", format_snr(parts_den["snr_db"]))

    st.markdown("**Numbers behind the SNR** (same clean reference for both)")
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
        download_button(noisy, f"task7_{source_basename(source)}_noisy.png")
    with c2:
        download_button(denoised, f"task7_{source_basename(source)}_denoised_{filter_name.split()[0].lower()}.png")


# ---------------------------------------------------------------------------
# Task 8 — blur & sharpening
# ---------------------------------------------------------------------------

KERNEL_LAPLACIAN_STR = "[ 0  -1   0 ]\n[ -1   4  -1 ]\n[ 0  -1   0 ]   (Laplacian, 4-neighbour)"
KERNEL_CUSTOM_STR = "[ 0  -1   0 ]\n[ -1   5  -1 ]   ← custom sharpening kernel (sum = 1)\n[ 0  -1   0 ]"


def page_t8() -> None:
    st.title("Task 8 · Detail Enhancement (Blur → Sharpening)")
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
        kernel_block = KERNEL_LAPLACIAN_STR + f"\n\nout = I - α·Laplacian(I),   α = {alpha}"
        explainer = (
            f"**Laplacian sharpening** with α = {alpha}: the 4-neighbour Laplacian is a "
            "second-derivative (band-pass) operator; it is negative where a pixel is brighter "
            "than its surroundings, so subtracting it from the image pushes those pixels even "
            "brighter — edges and fine detail get stronger. α = 0 gives the unsharpened blurred "
            "image; larger α increases edge emphasis (and ringing) at the cost of stability."
        )
    elif method.startswith("Custom"):
        intensity = st.slider("Kernel intensity (0.5 – 2)", 0.5, 2.0, 1.0, 0.1)
        sharpened = ops.sharpen_custom_kernel(blurred, intensity)
        kernel_block = KERNEL_CUSTOM_STR + f"\n\nout = blurred ⊗ (intensity · K),   intensity = {intensity}\n"
        explainer = (
            f"**Custom 3×3 kernel sharpening** with intensity {intensity}: the centre pixel is "
            "weighted +5 (or 5·intensity) while the four neighbours are subtracted (−1 each). "
            "The kernel sums to 1, so flat regions are unchanged and only local contrast around "
            "edges increases. This is the classic handcrafted sharpening convolution."
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

SOBEL_KX_STR = "Sobel x (horizontal gradient)      Sobel y (vertical gradient)\n\n  [ -2   0   2 ]                   [ -1  -2  -1 ]\n  [ -1   0   1 ]                   [  0   0   0 ]\n  [  2   0  -2 ]                   [  1   2   1 ]\n\n  gx = I ⊗ Kx      gy = I ⊗ Ky      magnitude = sqrt(gx² + gy²)"
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
    st.title("Task 9 · Edge Detection")
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
    st.divider()
    pages[nav_key]()


main()
