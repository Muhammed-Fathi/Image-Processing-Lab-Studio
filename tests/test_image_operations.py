"""Tests for the core image processing algorithms (src/image_operations.py).

Small NumPy-generated images are used so every check is exact and fast.
"""

import numpy as np
import pytest

from src import image_operations as ops


def one_pixel(rgb_values) -> np.ndarray:
    """A 1x1 RGB image with the given (R, G, B) values."""
    return np.array([[[int(v) for v in rgb_values]]], dtype=np.uint8)


def flat(h: int, w: int, value: int, channels: int = 1) -> np.ndarray:
    if channels == 1:
        return np.full((h, w), value, dtype=np.uint8)
    return np.full((h, w, channels), value, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Task 1 - channel separation
# ---------------------------------------------------------------------------

def test_bgr_to_rgb_swaps_first_and_last_channels():
    bgr = np.array([[[1, 2, 3]]], dtype=np.uint8)  # B=1, G=2, R=3
    rgb = ops.bgr_to_rgb(bgr)
    assert rgb[0, 0].tolist() == [3, 2, 1]  # now R=3, G=2, B=1


def test_separate_channels_returns_each_channel():
    rgb = one_pixel((10, 20, 30))
    r, g, b = ops.separate_channels(rgb)
    assert int(r[0, 0]) == 10
    assert int(g[0, 0]) == 20
    assert int(b[0, 0]) == 30


def test_isolate_channel_keeps_only_one_channel():
    rgb = one_pixel((10, 20, 30))
    for i in range(3):
        iso = ops.isolate_channel(rgb, i)
        expected = [0, 0, 0]
        expected[i] = [10, 20, 30][i]
        assert iso[0, 0].tolist() == expected



# ---------------------------------------------------------------------------
# Task 1 extension - HSV red color segmentation / color splash
# ---------------------------------------------------------------------------

def test_hsv_red_splash_preserves_matching_bgr_and_grays_other_pixels():
    import cv2

    bgr = np.array([[
        [20, 40, 230],     # saturated red: keep all three original BGR components
        [255, 0, 0],       # blue: convert to grayscale
        [0, 255, 0],       # green: convert to grayscale
        [255, 255, 255],   # white (S=0): do not select
    ]], dtype=np.uint8)
    original = bgr.copy()
    mask, splash = ops.hsv_red_color_splash(bgr)

    assert mask.dtype == np.uint8
    assert splash.dtype == np.uint8
    assert mask.shape == bgr.shape[:2]
    assert splash.shape == bgr.shape
    assert mask.tolist() == [[255, 0, 0, 0]]
    assert np.array_equal(splash[0, 0], bgr[0, 0])

    expected_gray = cv2.cvtColor(
        cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR
    )
    assert np.array_equal(splash[:, 1:], expected_gray[:, 1:])
    assert np.array_equal(bgr, original)  # processing must not mutate input


def test_hsv_red_splash_uses_both_red_hue_ranges_and_adjustable_sv():
    import cv2

    hsv = np.array([[
        [5, 200, 220],     # low red hue range
        [175, 200, 220],   # high red hue range (wrap-around)
        [60, 200, 220],    # non-red
        [5, 90, 220],      # muted red (excluded by S=100)
        [175, 200, 90],    # dark red (excluded by V=100)
    ]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    default_mask, _ = ops.hsv_red_color_splash(bgr)
    assert default_mask.tolist() == [[255, 255, 0, 0, 0]]

    relaxed_mask, _ = ops.hsv_red_color_splash(
        bgr, min_saturation=80, min_value=80
    )
    assert relaxed_mask.tolist() == [[255, 255, 0, 255, 255]]


def test_hsv_red_splash_accepts_grayscale_without_mistaking_it_for_red():
    gray = np.array([[20, 120, 230]], dtype=np.uint8)
    mask, splash = ops.hsv_red_color_splash(gray)
    assert not mask.any()
    assert splash.shape == (1, 3, 3)
    assert np.array_equal(splash[:, :, 0], gray)
    assert np.array_equal(splash[:, :, 1], gray)
    assert np.array_equal(splash[:, :, 2], gray)


@pytest.mark.parametrize("kwargs", [
    {"min_saturation": -1},
    {"min_saturation": 256},
    {"min_value": -1},
    {"min_value": 256},
])
def test_hsv_red_splash_rejects_invalid_sv_limits(kwargs):
    with pytest.raises(ValueError):
        ops.hsv_red_color_splash(one_pixel((0, 0, 255)), **kwargs)


def test_hsv_red_splash_rejects_invalid_shape_or_dtype():
    with pytest.raises(ValueError):
        ops.hsv_red_color_splash(np.zeros((2, 2, 4), dtype=np.uint8))
    with pytest.raises(ValueError):
        ops.hsv_red_color_splash(np.zeros((2, 2, 3), dtype=np.float32))



@pytest.mark.parametrize("selected_color,expected_mask", [
    ("red", [255, 0, 0, 0]),
    ("green", [0, 255, 0, 0]),
    ("blue", [0, 0, 255, 0]),
])
def test_hsv_color_splash_selects_each_primary_color(selected_color, expected_mask):
    """Red, green, blue each get their own HSV mask, not RGB channel zeroing."""
    import cv2

    bgr = np.array([[
        [30, 50, 230],    # red-colored but retains some green and blue
        [0, 255, 0],      # green
        [255, 0, 0],      # blue
        [255, 255, 255],  # white, not any saturated color
    ]], dtype=np.uint8)
    original = bgr.copy()
    mask, output = ops.hsv_color_splash(bgr, selected_color)
    assert mask.tolist() == [expected_mask]
    assert np.array_equal(bgr, original)

    gray_bgr = cv2.cvtColor(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY),
                            cv2.COLOR_GRAY2BGR)
    for col, marked in enumerate(expected_mask):
        assert np.array_equal(
            output[0, col],
            bgr[0, col] if marked else gray_bgr[0, col],
        )


def test_hsv_green_and_blue_use_their_respective_hue_intervals():
    import cv2

    hsv = np.array([[
        [30, 255, 255],  # yellow, not green
        [60, 255, 255],  # green
        [90, 255, 255],  # cyan, not green or blue
        [120, 255, 255], # blue
        [145, 255, 255], # purple, not blue
    ]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    mask_green, _ = ops.hsv_color_splash(bgr, "green")
    mask_blue, _ = ops.hsv_color_splash(bgr, "blue")
    assert mask_green.tolist() == [[0, 255, 0, 0, 0]]
    assert mask_blue.tolist() == [[0, 0, 0, 255, 0]]


def test_hsv_color_splash_adjustable_saturation_value_for_green_and_blue():
    import cv2

    hsv = np.array([[
        [60, 90, 220],   # muted green: excluded with S=100
        [120, 200, 90],  # dark blue: excluded with V=100
    ]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    green_default, _ = ops.hsv_color_splash(bgr, "green")
    blue_default, _ = ops.hsv_color_splash(bgr, "blue")
    green_relaxed, _ = ops.hsv_color_splash(bgr, "green", min_saturation=80)
    blue_relaxed, _ = ops.hsv_color_splash(bgr, "blue", min_value=80)
    assert green_default.tolist() == [[0, 0]]
    assert blue_default.tolist() == [[0, 0]]
    assert green_relaxed.tolist() == [[255, 0]]
    assert blue_relaxed.tolist() == [[0, 255]]


def test_hsv_red_wrapper_still_matches_generalized_function():
    img = np.array([[[0, 0, 255], [0, 255, 0]]], dtype=np.uint8)
    old_mask, old_result = ops.hsv_red_color_splash(img, 80, 90)
    new_mask, new_result = ops.hsv_color_splash(img, "red", 80, 90)
    assert np.array_equal(old_mask, new_mask)
    assert np.array_equal(old_result, new_result)


@pytest.mark.parametrize("color", ["purple", "", 123, None])
def test_hsv_color_splash_rejects_unknown_color(color):
    with pytest.raises(ValueError):
        ops.hsv_color_splash(one_pixel((1, 2, 3)), color)


# ---------------------------------------------------------------------------
# Task 2 - weighted grayscale
# ---------------------------------------------------------------------------

def test_weighted_grayscale_formula():
    # Gray = 0.299*100 + 0.587*150 + 0.114*200 = 140.75 -> 140 (truncated)
    rgb = one_pixel((100, 150, 200))
    gray = ops.weighted_grayscale(rgb)
    assert int(gray[0, 0]) in (140, 141)


def test_weighted_grayscale_gray_input_is_identity():
    g = flat(4, 4, 100)
    out = ops.weighted_grayscale(g)
    assert np.array_equal(out, g)  # weights sum to 1.0


def test_weighted_grayscale_not_simple_average():
    # R and B equal, G different: weighted and average differ.
    rgb = one_pixel((200, 100, 200))
    # weighted: 0.299*200 + 0.587*100 + 0.114*200 = 59.8 + 58.7 + 22.8 = 141.3
    # average: (200+100+200)/3 = 166.7
    gray = int(ops.weighted_grayscale(rgb)[0, 0])
    assert abs(gray - 141) <= 1
    assert abs(gray - 167) > 5


# ---------------------------------------------------------------------------
# Task 3 - arithmetic operations
# ---------------------------------------------------------------------------

def test_addition_clips_instead_of_uint8_wrapping():
    img = flat(2, 2, 250)
    out = ops.arithmetic_operation(img, "add", 10)
    assert int(out[0, 0]) == 255  # NOT 4 (the uint8 wrap-around value)


def test_subtraction_clips_at_zero():
    img = flat(2, 2, 5)
    out = ops.arithmetic_operation(img, "sub", 10)
    assert int(out[0, 0]) == 0


def test_multiplication_scales():
    img = flat(2, 2, 100)
    out = ops.arithmetic_operation(img, "mul", 0.5)
    assert int(out[0, 0]) == 50


def test_division_and_zero_guard():
    img = flat(2, 2, 100)
    out = ops.arithmetic_operation(img, "div", 2.0)
    assert int(out[0, 0]) == 50
    # C = 0 must not crash: it is clamped away from zero -> 100/0.05 clips to 255
    out0 = ops.arithmetic_operation(img, "div", 0.0)
    assert int(out0[0, 0]) == 255


def test_arithmetic_unknown_operation_raises():
    with pytest.raises(ValueError):
        ops.arithmetic_operation(flat(2, 2, 10), "xor", 1)


# ---------------------------------------------------------------------------
# Task 4 - negative and solarization
# ---------------------------------------------------------------------------

def test_negative_is_full_complement():
    img = np.array([[0, 10, 128, 255]], dtype=np.uint8)
    assert ops.negative(img).tolist() == [[255, 245, 127, 0]]


def test_solarize_inverts_only_above_threshold():
    img = np.array([[50, 126, 127, 200]], dtype=np.uint8)
    out = ops.solarize(img, threshold=127)
    assert out.tolist() == [[50, 126, 128, 55]]  # 255-127=128, 255-200=55


def test_solarize_threshold_zero_equals_negative():
    img = np.arange(256, dtype=np.uint8).reshape(1, 256)
    assert np.array_equal(ops.solarize(img, 0), ops.negative(img))


# ---------------------------------------------------------------------------
# Task 5 - channel elimination
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("channel,expected", [
    (0, [0, 20, 30]),
    (1, [10, 0, 30]),
    (2, [10, 20, 0]),
])
def test_eliminate_channel_zeros_only_the_selected_channel(channel, expected):
    rgb = one_pixel((10, 20, 30))
    out = ops.eliminate_channel(rgb, channel)
    assert out[0, 0].tolist() == expected


def test_eliminate_channel_invalid_channel_raises():
    with pytest.raises(ValueError):
        ops.eliminate_channel(one_pixel((1, 2, 3)), 3)


# ---------------------------------------------------------------------------
# Task 6 - histogram enhancement
# ---------------------------------------------------------------------------

def test_stretch_maps_range_to_full_scale():
    img = np.array([[100, 125, 150]], dtype=np.uint8)
    out, lo, hi = ops.histogram_stretch(img)
    assert (lo, hi) == (100, 150)
    assert int(out.min()) == 0
    assert int(out.max()) == 255
    # Middle of the input range maps to the middle of the output range
    assert abs(int(out[0, 1]) - 127) <= 1


def test_stretch_constant_image_does_not_crash():
    img = flat(4, 4, 128)
    out, lo, hi = ops.histogram_stretch(img)
    assert lo == hi == 128
    assert np.array_equal(out, img)  # returned unchanged


def test_equalization_spreads_skewed_histogram():
    img = np.zeros((32, 32), dtype=np.uint8)
    img[:, :30] = 5  # almost everything dark, a few bright pixels
    img[:, 30] = 255
    out = ops.histogram_equalization(img)
    assert out.dtype == np.uint8
    assert out.shape == img.shape
    assert int(out.max()) >= 250  # brightest pixel pushed to the top of the scale
    assert int(out.min()) == 0


def test_equalization_color_uses_y_channel_and_preserves_shape():
    img = flat(8, 8, 40, channels=3)
    img[:, :, 2] = 220  # strong red tint
    out = ops.histogram_equalization(img)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_make_low_contrast_compresses_range():
    img = (np.random.default_rng(0).random((16, 16)) * 255).astype(np.uint8)
    out = ops.make_low_contrast(img, band=(105, 165))
    assert 100 <= int(out.min())
    assert int(out.max()) <= 170


# ---------------------------------------------------------------------------
# Task 7 - noise and filters
# ---------------------------------------------------------------------------

def test_salt_pepper_fraction_and_determinism():
    base = flat(50, 50, 128)  # mid-gray so corrupted pixels are countable
    out1 = ops.add_salt_and_pepper(base, 0.25, seed=7)
    out2 = ops.add_salt_and_pepper(base, 0.25, seed=7)
    out3 = ops.add_salt_and_pepper(base, 0.25, seed=8)
    assert np.array_equal(out1, out2)      # same seed -> reproducible
    assert not np.array_equal(out1, out3)  # different seed -> different noise
    corrupted = np.isin(out1, [0, 255]).mean()
    assert abs(corrupted - 0.25) < 0.01


def test_salt_pepper_zero_ratio_is_identity():
    base = flat(50, 50, 128)
    assert np.array_equal(ops.add_salt_and_pepper(base, 0.0, seed=1), base)


# ---------------------------------------------------------------------------
# REGRESSION (audit issue 4): S&P corrupts whole PIXELS, not channel elements
# ---------------------------------------------------------------------------

def test_salt_pepper_corrupts_exactly_ratio_fraction_of_pixels_color():
    base = np.full((100, 100, 3), 128, dtype=np.uint8)
    out = ops.add_salt_and_pepper(base, 0.10, seed=11)
    diff = out.astype(np.int16) != 128
    pixel_mask = diff.any(axis=2)
    # exactly round(0.10 * 100 * 100) = 1000 corrupted PIXELS (not ~2700 elements)
    assert int(pixel_mask.sum()) == 1000
    ys, xs = np.where(pixel_mask)
    values = out[ys, xs]
    # every corrupted pixel is full black or full white on ALL channels
    assert ((values == 0).all(axis=1) | (values == 255).all(axis=1)).all()
    # uncorrupted pixels are untouched
    assert (out[~pixel_mask] == 128).all()


def test_salt_pepper_grayscale_whole_pixels():
    base = np.full((10, 10), 128, dtype=np.uint8)
    out = ops.add_salt_and_pepper(base, 0.20, seed=5)
    corrupted = out != 128
    assert int(corrupted.sum()) == 20  # round(0.20 * 100) pixels
    assert set(np.unique(out)).issubset({0, 128, 255})


def test_salt_pepper_ratio_uses_h_w_not_flattened_channels():
    """10x10x3 with ratio 0.5 must corrupt 50 pixels (150 channel
    elements), i.e. the count is based on H*W."""
    base = np.full((10, 10, 3), 128, dtype=np.uint8)
    out = ops.add_salt_and_pepper(base, 0.5, seed=2)
    assert int((out.astype(np.int16) != 128).any(axis=2).sum()) == 50


def test_gaussian_noise_std_and_determinism():
    base = flat(60, 60, 128)
    out1 = ops.add_gaussian_noise(base, 20, seed=3)
    out2 = ops.add_gaussian_noise(base, 20, seed=3)
    assert np.array_equal(out1, out2)
    diff = (out1.astype(np.float64) - base.astype(np.float64))
    assert 15 < diff.std() < 25  # close to sigma (clip truncation aside)


def test_median_filter_removes_single_impulse():
    img = flat(8, 8, 128)
    img[4, 4] = 255  # one salt pixel
    out = ops.median_filter(img, 3)
    assert int(out[4, 4]) == 128  # impulse removed
    assert (out == 128).all()     # flat region stays flat


def test_gaussian_and_mean_filters_valid():
    img = flat(8, 8, 100)
    for f in (ops.gaussian_filter, ops.mean_filter):
        out = f(img, 3)
        assert out.shape == img.shape
        assert out.dtype == np.uint8
        assert (out == 100).all()  # constant image stays constant


def test_filter_functions_accept_even_kernel_gracefully():
    img = flat(8, 8, 100)
    out = ops.median_filter(img, 4)  # auto-corrected to an odd kernel
    assert out.shape == img.shape


# ---------------------------------------------------------------------------
# Task 8 - blur and sharpening
# ---------------------------------------------------------------------------

def test_blur_smooths_a_step_edge():
    img = flat(8, 16, 0)
    img[:, 8:] = 255  # hard vertical step
    out = ops.gaussian_blur(img, 5)
    # Blurring must create intermediate values around the step
    assert (out[:, 7] > 0).any() and (out[:, 9] < 255).any()
    assert out.dtype == np.uint8


def test_sharpen_flat_image_is_unchanged():
    img = flat(8, 8, 128)
    assert (ops.sharpen_laplacian(img, 1.0) == 128).all()
    assert (ops.sharpen_custom_kernel(img, 1.0) == 128).all()
    assert (ops.unsharp_mask(img, 1.0, 5) == 128).all()  # kernel sums to 1


# ---------------------------------------------------------------------------
# REGRESSION (audit issue 1): Laplacian sharpening sign / depth
# ---------------------------------------------------------------------------

def test_laplacian_sharpen_bright_feature_is_not_erased():
    """Black 5x5 image with a bright centre pixel: with the +4-centre
    Laplacian the response at the centre is +4*100, so out = I + 1*L must
    give a centre >= 100 (it saturates to 255). The old code (uint8
    convolution + subtraction) produced 0 here, erasing the feature."""
    img = np.zeros((5, 5), dtype=np.uint8)
    img[2, 2] = 100
    out = ops.sharpen_laplacian(img, 1.0)
    assert int(out[2, 2]) >= 100


def test_laplacian_sharpen_alpha_zero_is_exact_identity():
    img = flat(5, 5, 100)
    img[2, 2] = 10
    assert np.array_equal(ops.sharpen_laplacian(img, 0.0), img)


def test_laplacian_sharpen_dark_feature_gets_darker():
    """Mirror case: a dark pixel on a bright background must get darker
    (the signed float response is negative there), proving the sign is
    correct in both directions."""
    img = np.full((5, 5), 200, dtype=np.uint8)
    img[2, 2] = 100
    out = ops.sharpen_laplacian(img, 1.0)
    assert int(out[2, 2]) < 100


# ---------------------------------------------------------------------------
# REGRESSION (audit issue 7): custom kernel DC gain
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("s", [0.5, 1.0, 1.5])
def test_custom_kernel_flat_constant_stays_constant_at_any_strength(s):
    """K(s) = I3 + s*(K_base - I3) has sum 1 for every s, so a flat
    constant image must be reproduced EXACTLY (uint8 equality)."""
    img = flat(10, 10, 100)
    assert (ops.sharpen_custom_kernel(img, s) == 100).all()


def test_custom_kernel_strength_changes_only_high_pass():
    """On a flat-100 image with one bright pixel, strength 1.5 must
    sharpen (centre >= strength 0.5's centre) while flat neighbours
    remain 100 (no DC gain at any strength)."""
    img = np.full((10, 10), 100, dtype=np.uint8)
    img[5, 5] = 200
    out_half = ops.sharpen_custom_kernel(img, 0.5)
    out_full = ops.sharpen_custom_kernel(img, 1.5)
    # Flat corner region must be untouched at both strengths
    assert (out_half[:2, :2] == 100).all() and (out_full[:2, :2] == 100).all()
    # Stronger strength -> stronger emphasis of the bright centre
    assert int(out_full[5, 5]) >= int(out_half[5, 5]) > 100


def test_custom_kernel_effective_kernel_sum_is_one():
    """Rebuild the documented K(s) = delta + s*(K_base - delta) directly
    (delta = unit impulse, sum 1) and check the DC property sum = 1."""
    for s in (0.3, 1.0, 2.0):
        k = ops.DELTA_3X3 + s * (ops.SHARPEN_CUSTOM_KERNEL - ops.DELTA_3X3)
        assert abs(float(k.sum()) - 1.0) < 1e-6
    # s = 1 reproduces the base kernel exactly
    k1 = ops.DELTA_3X3 + 1.0 * (ops.SHARPEN_CUSTOM_KERNEL - ops.DELTA_3X3)
    assert np.array_equal(k1, ops.SHARPEN_CUSTOM_KERNEL)


def test_sharpen_outputs_valid_uint8():
    rng = np.random.default_rng(0)
    img = (rng.random((16, 16)) * 255).astype(np.uint8)
    blurred = ops.gaussian_blur(img, 5)
    for out in (
        ops.sharpen_laplacian(blurred, 1.5),
        ops.sharpen_custom_kernel(blurred, 1.5),
        ops.unsharp_mask(blurred, 1.5, 5),
    ):
        assert out.shape == img.shape
        assert out.dtype == np.uint8
        assert int(out.min()) >= 0 and int(out.max()) <= 255


# ---------------------------------------------------------------------------
# Task 9 - edge detection
# ---------------------------------------------------------------------------

def test_sobel_detects_vertical_step():
    img = flat(16, 32, 0)
    img[:, 16:] = 255
    edges = ops.sobel_edges(img, 3)
    assert edges.dtype == np.uint8
    # The vertical step must produce strong horizontal-gradient response
    assert int(edges[:, 14:18].max()) > 100
    # Flat far-left region must be (almost) edge-free
    assert int(edges[:, :6].max()) < 30


def test_sobel_constant_image_is_all_zero():
    edges = ops.sobel_edges(flat(8, 8, 100), 3)
    assert (edges == 0).all()


def test_canny_returns_binary_edges():
    img = flat(16, 32, 0)
    img[:, 16:] = 255
    edges = ops.canny_edges(img, 50, 150)
    assert set(np.unique(edges)).issubset({0, 255})
    assert (edges == 255).any()


def test_canny_thresholds_respected_low_high_order():
    img = flat(16, 32, 0)
    img[:, 16:] = 255
    edges = ops.canny_edges(img, 150, 50)  # reversed thresholds must not crash
    assert edges.shape == img.shape


def test_laplacian_edges_detect_step():
    img = flat(16, 32, 0)
    img[:, 16:] = 255
    edges = ops.laplacian_edges(img)
    assert int(edges.max()) > 0
    assert edges.dtype == np.uint8
