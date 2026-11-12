# Image Processing Lab Studio

An interactive, educational **Streamlit** application that implements **nine university
image-processing tasks** across four labs. It is built for a practical evaluation and an
oral discussion: every task lets you pick parameters, compare the original and processed
images, inspect the numbers (statistics, histograms, SNR, matrix excerpts), read the exact
mathematics, and review discussion notes that justify the technique chosen.

> Educational tool. Classical, deterministic image processing only — no deep learning, no
> external APIs, no internet access at runtime, no medical-grade claims.

---

## 1. Project overview

- One Streamlit app, one entry point (`app.py`), plain Python modules for all algorithms.
- Upload your own image **or** use one of the 7 bundled demonstration images (generated
  by this project, reproducible, no licenses to track).
- Consistent task layout: objective → image source → parameters → comparisons → numerical
  evidence → formula → *why this technique* → discussion notes → download.

## 2. University project objectives

| # | Objective (as required) | Where |
|---|-------------------------|-------|
| 1 | RGB channel separation with correct BGR↔RGB handling | Task 1 |
| 2 | Grayscale via weighted luminance formula (averaging/mean/median **prohibited**) | Task 2 |
| 3 | Arithmetic enhancement: choose the operation that fits a flawed image | Task 3 |
| 4 | Complement (negative) or solarization, with the exact rule explained | Task 4 |
| 5 | Examine the RGB matrix and eliminate exactly one channel | Task 5 |
| 6 | Histogram of a low-contrast image, then stretching or equalization | Task 6 |
| 7 | Inject artificial noise, remove it with a suitable filter, compute **SNR from scratch** | Task 7 |
| 8 | Blur an image deliberately, then sharpen it | Task 8 |
| 9 | Select an appropriate edge operator and reveal structural boundaries | Task 9 |

## 3. Features

- Sidebar navigation for the dashboard and all 9 tasks (plus quick navigation from the dashboard).
- Image upload (PNG/JPG/BMP/TIFF/WebP) with Pillow + OpenCV double validation, or bundled samples.
- Interactive sliders/selections for every parameter (constants, thresholds, kernels, noise, seed).
- Original-vs-processed comparisons everywhere.
- Live numerical evidence: intensity statistics tables, 256-bin Matplotlib histograms,
  pixel inspectors, 5×5 RGB matrix excerpts, SNR with its intermediate powers.
- Formula boxes showing the exact operation applied, generated from the current parameters.
- Discussion notes (expander) on every page, written for oral evaluation and based on the
  actual image statistics when possible.
- PNG download of processed results.
- Minimal, emoji-free dashboard with clean lab cards, compact navigation, accessible image-source controls, and responsive two/three-column layouts.

## 4. The nine tasks (by lab)

### Lab 1 — Image Fundamentals
1. **RGB Channel Separation** — BGR vs RGB demonstration, three channel intensity maps,
   three color-isolated images, a pixel inspector, and a 5×5 RGB matrix excerpt.
2. **Custom Grayscale** — `Gray = 0.299R + 0.587G + 0.114B` implemented manually in
   NumPy (float → clip → uint8). A side-by-side with OpenCV's own conversion is shown
   *only as a reference check*; the conversion itself never uses `cv2.cvtColor`.

### Lab 2 — Pixel & Intensity Operations
3. **Arithmetic Operations** — `I+C`, `I−C`, `I×C`, `I÷C` in float with clipping,
   a division-by-zero guard, measured exposure analysis of the input image, and
   evidence-based (not absolute) guidance for choosing the operation.
4. **Negative / Solarization** — full complement `255−I` and threshold-controlled
   partial inversion with the exact rule and a live percentage of affected pixels.
5. **Channel Elimination** — per-channel statistics, dominant-channel detection with
   color-cast guidance, a matrix excerpt, and elimination of exactly one channel.

### Lab 3 — Histogram Enhancement
6. **Histogram Enhancement** — real 256-bin histograms before/after (computed from the
   actual data), linear **stretching** (with the zero-range guard) and **equalization**
   (on the Y channel of YCrCb for color images), optional brightness fine-tuning, and a
   demo helper that compresses any image into a narrow band to create a low-contrast input.

### Labs 4 & 5 — Filtering & Edge Detection
7. **Noise Reduction + SNR** — salt-and-pepper or Gaussian noise (seeded, reproducible),
   Median / Gaussian / Mean filters with a recommendation, and **SNR computed from
   scratch** before and after denoising (see §11).
8. **Blur & Sharpening** — deliberate Gaussian blur, then Laplacian-based sharpening,
   a custom 3×3 kernel, or unsharp masking; honest note that blur destroys information
   and sharpening cannot restore it.
9. **Edge Detection** — Sobel (normalized magnitude, so negative gradients are not
   truncated), Canny (low/high thresholds, hysteresis explained), and Laplacian.

## 5. Technologies used

| Technology | Role |
|------------|------|
| Python 3.10+ | language |
| Streamlit | interactive UI |
| OpenCV (`opencv-python-headless`) | image I/O, filters, Canny, equalization, color conversion |
| NumPy | all pixel-level math (float internally, clipped to uint8) |
| Matplotlib | histograms and plots |
| Pillow | independent validation of uploaded files |
| Pytest | automated tests (development only) |

Pandas is intentionally **not** used: `st.table` renders plain lists of dictionaries,
which keeps the dependency list minimal.

## 6. Folder structure

```
Image-Processing-Lab-Studio/
├── app.py                    # Streamlit application (dashboard + 9 task pages)
├── src/
│   ├── __init__.py
│   ├── image_operations.py   # all 9 task algorithms (pure NumPy/OpenCV)
│   ├── image_utils.py        # loading/saving, sample registry, validation
│   └── metrics.py            # SNR from scratch + intensity statistics
├── assets/
│   └── samples/              # 7 bundled demonstration images (see §10)
├── scripts/
│   └── generate_samples.py   # regenerates the demo images (deterministic)
├── tests/
│   ├── test_image_operations.py
│   ├── test_metrics.py
│   ├── test_image_utils.py
│   ├── test_assets.py
│   └── test_app.py
├── conftest.py               # makes the project root importable for tests
├── .streamlit/
│   └── config.toml           # light theme configuration
├── requirements.txt          # runtime dependencies
├── requirements-dev.txt      # runtime + pytest
├── README.md
└── .gitignore
```

## 7. Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Activate it:

- **Windows (PowerShell):** `.venv\Scripts\Activate.ps1`
- **Windows (cmd):** `.venv\Scripts\activate.bat`
- **Linux / macOS:** `source .venv/bin/activate`

Install the runtime dependencies:

```bash
pip install -r requirements.txt
```

(Optionally, for development and testing: `pip install -r requirements-dev.txt`)

## 8. Running the application

```bash
streamlit run app.py
```

Streamlit opens the app in your browser (default: http://localhost:8501). To regenerate
the bundled sample images at any time (e.g. after moving the repository):

```bash
python scripts/generate_samples.py
```

## 9. How to use the interface

1. **Dashboard** — project overview, lab cards, task checklist, quick navigation.
2. Pick a task in the **sidebar** (or the quick-navigation box on the dashboard).
3. On each task page, choose a **demonstration image** or **upload your own**
   (PNG, JPG, BMP, TIFF, WebP).
4. Adjust the **parameters** (sliders, selectors).
5. Read the **comparisons**, **statistics**, **histograms** and **formulas**.
6. Open the **Discussion notes** expander before your oral evaluation.
7. **Download** the processed image(s) as PNG.

## 10. Bundled demonstration images

All seven images are **generated deterministically by `scripts/generate_samples.py`**
(pure NumPy/OpenCV drawing code, no random components) and are owned by this project —
no third-party photography, no licenses to track, and no runtime downloads.

| File | Content / purpose |
|------|-------------------|
| `color_scene.png` | Synthetic color landscape — general RGB and grayscale demos (Tasks 1–2) |
| `dark_scene.png` | Underexposed version of the scene — arithmetic enhancement demo (Task 3) |
| `midtone_gradient.png` | Smooth mid-tone gradient with shapes — negative / solarization demo (Task 4) |
| `color_cast.png` | Scene with a strong warm color cast — channel elimination demo (Task 5) |
| `low_contrast.png` | Scene compressed into a narrow intensity band (≈105–165) — histogram demo (Task 6) |
| `detail_scene.png` | Scene with fine line detail — noise and sharpening demo, the clean reference (Tasks 7–8) |
| `geometric_edges.png` | High-contrast geometric shapes and text — edge detection demo (Task 9) |

## 11. Important mathematical formulas

- **RGB → grayscale (ITU-R BT.601):** `Gray = 0.299·R + 0.587·G + 0.114·B`
- **Arithmetic:** `I_out = clip(I ⊕ C, 0, 255)` with ⊕ ∈ {+, −, ×, ÷} (float, then clip)
- **Negative:** `out = 255 − I`
- **Solarization:** `out = 255 − I if I ≥ T else I`
- **Channel elimination:** e.g. remove Red: `(R, G, B) → (0, G, B)`
- **Histogram stretching:** `out = (I − I_min) · 255 / (I_max − I_min)` (skipped if range = 0)
- **Histogram equalization:** `out(v) = round(255 · (CDF(v) − CDF_min) / (N − CDF_min))`
- **Gaussian blur / sharpening kernels:**
  Laplacian `L = [[0,−1,0],[−1,4,−1],[0,−1,0]]` (computed in float32, then
  `out = I + α·(I ⊗ L)` — the signed response is *added*, positive where a pixel
  is brighter than its neighbours);
  custom `K_base = [[0,−1,0],[−1,5,−1],[0,−1,0]]` with effective kernel
  `K(s) = δ + s·(K_base − δ)` (δ = unit impulse, so `sum K(s) = 1` at any
  strength and flat regions stay constant);
  unsharp mask `out = I + k·(I − blurred(I))`
- **Sobel magnitude:** `√(gx² + gy²)` with `gx = I ⊗ Kx`, `gy = I ⊗ Ky`

### SNR calculation (implemented from scratch in `src/metrics.py`)

For a clean reference image **S** and an evaluated image **X** (the noisy image or the
denoised image — always the *same* S):

```
Signal power = Σ S²          (sum of squared clean pixels, float64)
Noise power  = Σ (S − X)²    (sum of squared errors)
SNR (dB)     = 10 · log10( Signal power / Noise power )
```

- No library PSNR/SNR function is used; the app shows the two powers and the result.
- If `X == S` exactly, the noise power is zero and the SNR is reported as **∞**.
- If the signal has zero energy but the error is non-zero, the ratio is 0 and
  `10·log10(0) = −∞`, so the SNR is reported as **−∞** (deliberately not a fake 0 dB).
- This is a **power-ratio SNR, not PSNR**: PSNR would use the MSE and reference 255².
  The two quantities are different and must not be conflated in the discussion.
- A higher SNR (under this definition) means less error relative to the reference signal.

## 12. Image input requirements

- Formats: PNG, JPG/JPEG, BMP, TIFF, WebP.
- 8-bit images; grayscale (1-channel) or color (3-channel; 4-channel BGRA is
  converted to BGR, dropping the alpha channel).
- Grayscale uploads are promoted to 3 channels where a color image is required (the app
  tells you when this happens).
- Unreadable or corrupt files are rejected with a clear error message — the app never
  crashes on a bad upload.
- No images are downloaded from the internet at runtime.

## 13. Short algorithm explanations

- **Channel separation** splits the stored color matrix into its three primary components;
  OpenCV's BGR order is converted to RGB before display/analysis.
- **Weighted grayscale** projects color onto perceived luminance; green dominates because
  the human eye is most sensitive to it under normal lighting.
- **Arithmetic operations** are point operations: addition/subtraction shift the whole
  tonal curve, multiplication/division scale it.
- **Negative / solarization** invert tonal values, fully or only above a threshold.
- **Channel elimination** shows what a single primary contributes by zeroing it.
- **Stretching** linearly remaps the observed intensity range to 0–255; **equalization**
  redistributes values via the histogram's CDF. Color images are processed on the Y
  (luminance) channel of YCrCb to preserve hue.
- **Salt-and-pepper noise** replaces whole pixels — all channels at once — with pure
  black or white (seeded, reproducible); **median filtering** rejects such impulse
  outliers; **Gaussian/mean smoothing** averages out additive Gaussian noise.
- **Sharpening** re-emphasizes high frequencies (second derivative or high-pass detail);
  it cannot recover frequencies removed by the blur.
- **Sobel** measures gradient strength; **Canny** is a 5-stage detector producing thin,
  hysteresis-filtered edges; **Laplacian** is an isotropic second-derivative response.

## 14. Example demonstrations (ready-made for the oral exam)

1. *Task 3:* open with the bundled **dark scene** → the exposure banner reports "dark"
   → choose **Addition** → show the mean rising and the surviving shadow content
   becoming visible (values clipped to 0 at capture stay lost — say so).
2. *Task 6:* open with the bundled **low-contrast scene** → narrow histogram → choose
   **Stretching** → the histogram spreads to the full range; switch to **Equalization**
   and compare the histograms and statistics.
3. *Task 7:* keep the seed at 42, salt-and-pepper 8 %, recommend **Median** → show the
   SNR table (noisy vs denoised) — the denoised SNR is higher.
4. *Task 9:* on the **geometric edges** sample, compare Sobel (thick bands) vs Canny
   (thin 1-px boundaries) and cite which is better for segmentation.

## 15. Testing

```bash
pip install -r requirements-dev.txt   # installs pytest as well
python -m pytest tests/ -v
```

The suite covers: channel extraction, weighted grayscale, arithmetic clipping,
negative/solarization, channel elimination, stretching (incl. constant images),
noise generation (determinism + intensity), median/Gaussian/mean filters, SNR
(incl. zero-noise and known-value cases), sharpening validity, edge detectors,
image loading/validation utilities, the bundled assets, `requirements.txt`
completeness, and the Streamlit app itself (every page executed through
`streamlit.testing.v1.AppTest`, asserting no exceptions and that task pages render
images).

## 16. Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'streamlit'` | Activate the venv and run `pip install -r requirements.txt` |
| `cv2` import errors after installing both `opencv-python` and `opencv-python-headless` | Keep only one: `pip uninstall -y opencv-python opencv-python-headless && pip install opencv-python-headless` |
| High-DPI blurry plots on some systems | Matplotlib figures are fixed-size; no action needed |
| Large photos feel slow | Processing is O(pixels); try images under ~4000 px per side for snappier sliders |
| Samples missing after a checkout | `python scripts/generate_samples.py` |
| Port 8501 already in use | `streamlit run app.py --server.port 8502` |

## 17. Deployment

### GitHub

```bash
git init
git add .
git commit -m "Image Processing Lab Studio: 9 university tasks in one Streamlit app"
git branch -M main
git remote add origin https://github.com/<your-username>/Image-Processing-Lab-Studio.git
git push -u origin main
```

### Streamlit Community Cloud

1. Create a free account at https://streamlit.io/cloud and connect the GitHub repository.
2. New **Deploy → New app** with:
   - Main file path: `app.py`
   - Python version: 3.10 or newer
   - `requirements.txt` is detected automatically.
3. Deploy. The app uses only local assets, so no secrets or environment variables are needed.

## 18. Limitations and important notes

- All algorithms are classical and deterministic; results are exact, but perceptual
  quality (e.g. sharpening) is a trade-off, not a guarantee.
- Sharpening cannot reconstruct information removed by blur; denoising always softens
  some real detail. The app states this on the relevant pages.
- SNR as implemented is a power ratio; do not quote it as PSNR.
- Histogram enhancement on color images works on luminance (Y) — hue is preserved, which
  is a deliberate, sensible choice rather than per-channel equalization.
- No resizing of user images is performed; very large photographs may be slower.
- This is an educational tool, not a production imaging pipeline, and it makes no
  medical or professional-grade claims.

## 19. License

The code in this repository is provided for academic use. The bundled sample images are
generated by the project itself.
