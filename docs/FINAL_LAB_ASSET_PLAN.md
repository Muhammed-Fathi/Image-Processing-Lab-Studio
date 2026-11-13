# Final lab image set and oral-exam plan

This document records the nine **approved exercises**, the six source photographs,
the one derived low-contrast input, and the reproducible image-preparation procedure.

## How to install the photographs

From the repository root, with the virtual environment active:

```bat
.venvScriptspython.exe scriptsprepare_final_assets.py --download
```

If Wikimedia requests fail (or if you want to use the exact photographs downloaded
during the review), put **the six unprocessed originals** in one dedicated folder
with filenames that include the keywords `poppies`, `floral`, `sunset`,
`compass`, `rock`, and `museum`. Then run:

```bat
.venvScriptspython.exe scriptsprepare_final_assets.py --source-dir "D:PathToSix_Photos"
```

The script saves the six selected photographs (normalized size, JPEG) and
creates **rock_low_contrast.png** using the project's existing
`make_low_contrast(..., band=(105, 165))`. It also creates
`assets/final_assets_manifest.json` with SHA-256 checksums.

**Do not** provide the processed Task 3/4/5/7/8/9 outputs as the script's
inputs; the preparation script requires the original photograph in each case.
Do not place duplicate copies in the input folder.

Once installed, each Streamlit page prefers its approved photograph and parameter
defaults. If they are not installed yet, the existing seven synthetic samples
continue to work, so the app and unit tests remain usable offline.

## Approved tasks

| Task | Photo or working input | Selected settings | Interpretation |
|---|---|---|---|
| 1 | `poppies_rgb_grayscale.jpg` | RGB channel separation | Compare intensity maps and isolated colors |
| 2 | `poppies_rgb_grayscale.jpg` | 0.299R + 0.587G + 0.114B | Implemented manually, not simple average |
| 3 | `dark_floral_tones.jpg` | Multiplication, C=1.8 | Increases contrast in a dark scene; some highlight clipping |
| 4 | `sunset_solarization.jpg` | Solarization, threshold 170 | Inverts stored channel values >=170 |
| 5 | `blue_cast_compass.jpg` | Eliminate Blue (B=0) | Diagnostic visual comparison, **not** color correction |
| 6 | `rock_low_contrast.png` | Histogram stretching, brightness 0, helper OFF | Prepared by compressing the real rock photo first |
| 7 | `museum_clean.jpg` | Salt-and-pepper 8%, seed 42, Median k=3 | Observed SNR 13.03 -> 20.04 dB (+7.01 dB) on review image |
| 8 | `museum_clean.jpg` | Gaussian blur k=7, Custom 3x3 kernel, strength=1.0 | Sharpened edges, but did not reconstruct all lost information |
| 9 | `blue_cast_compass.jpg` | Canny low=50, high=150 | Clear drafting-compass contours against plain background |

The SNR figures above were observed in the user's image comparison; they
may change slightly if the source image bytes are resized or recompressed.
**Always use the currently measured Streamlit values in the final oral report**.

## Image sources and terms

Photographs were selected from Wikimedia Commons:

| Asset | Photographer | License | Source |
|---|---|---|---|
| `poppies_rgb_grayscale.jpg` | corina ardeleanu | CC0 1.0 | https://commons.wikimedia.org/wiki/File:Red_poppies_under_morning_sun_(Unsplash).jpg |
| `dark_floral_tones.jpg` | Annie Spratt | CC0 1.0 | https://commons.wikimedia.org/wiki/File:Dark_Floral_Tones_(Unsplash).jpg |
| `sunset_solarization.jpg` | Jagyasini Malakar | CC0 1.0 | https://commons.wikimedia.org/wiki/File:Tree_silhouette_sunset._(Unsplash).jpg |
| `blue_cast_compass.jpg` | Fructibus | CC0 1.0 | https://commons.wikimedia.org/wiki/File:Blue_color_cast_-_Compass_1.jpg |
| `rock_original.jpg` | Gaius Cornelius | **CC BY-SA 4.0** | https://commons.wikimedia.org/wiki/File:Rock_1,_Elizabeth_Castle,_Jersey.jpg |
| `museum_clean.jpg` | Scott Webb | CC0 1.0 | https://commons.wikimedia.org/wiki/File:Colorful_%22Museum%E2%80%9D_sign_on_landscape_(Unsplash).jpg |

**Rock derivative**: `rock_low_contrast.png` was adapted from
`Rock 1, Elizabeth Castle, Jersey.jpg`, by **Gaius Cornelius**,
under **Creative Commons Attribution-ShareAlike 4.0 International**.
Changes made: resized to fit the project and contrast-compressed
with `make_low_contrast(..., band=(105, 165))`.
Both the source and modified rock image remain under CC BY-SA 4.0.
License: https://creativecommons.org/licenses/by-sa/4.0/

The Commons file descriptions, rather than generic Wikimedia site terms,
are the source of each individual file license. Retain this attribution and
modification note in any distributed report.

## Final quality checklist

- [ ] Run `prepare_final_assets.py` and verify seven image files plus JSON manifest.
- [ ] Open Task 6 using `rock_low_contrast.png` with Demo Helper **OFF**.
- [ ] Check all 9 pages and confirm approved initial methods/parameters.
- [ ] Run `python -m pytest tests/ -q` in the project environment.
- [ ] Inspect original, processed and metric displays; export the outputs needed for the PDF.
- [ ] Commit the new photos, scripts, data manifest, attribution and code together.
- [ ] Merge the asset PR only after tests and visual QA pass.
- [ ] Final English PDF should explain formulas, comparison, reasons and limitations
  and quote post-installation metrics from the tested photo bytes.
