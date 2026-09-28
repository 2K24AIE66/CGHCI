
# Name: Tooba
# Roll number: 2K24/AIE/66

"""
Perceptual Computing: Graphics, Vision, and Human Interaction
Lecture 1 - Python Implementations

Covers every concept/worked example from the slides:
 1. Total Pixel Count (Resolution)
 2. Pixels Per Inch (PPI)
 3. Aspect Ratio
 4. Color Depth (bits -> number of colors)
 5. Frame Buffer Size (memory footprint)
 6. Binary Thresholding (Monochrome mapping)
 7. Pixel Channel Bitmasking (RGB channel extraction)
 8. Downsampling (stride-based spatial scaling)
 9. Video Throughput (bitrate calculation)
10. Compression Ratio synthesis problem

Requires: numpy
Run:  python perceptual_computing_lecture1.py
"""

import math
import numpy as np


# ---------------------------------------------------------------------------
# 1. Total Pixel Count  (Slide: "Resolution: Count vs. Dimension")
# ---------------------------------------------------------------------------
def total_pixel_count(width: int, height: int) -> int:
    """Total Pixels = W x H"""
    return width * height


# ---------------------------------------------------------------------------
# 2. Pixels Per Inch (PPI)  (Slide: "Pixels Per Inch (PPI)")
#    PPI = sqrt(w_p^2 + h_p^2) / d_i
# ---------------------------------------------------------------------------
def calculate_ppi(width_px: int, height_px: int, diagonal_inches: float) -> float:
    diagonal_px = math.sqrt(width_px ** 2 + height_px ** 2)
    return diagonal_px / diagonal_inches


# ---------------------------------------------------------------------------
# 3. Aspect Ratio  (Slide: "Aspect Ratio & Framing Metrics")
# ---------------------------------------------------------------------------
def aspect_ratio(width: int, height: int) -> float:
    return width / height


def simplify_aspect_ratio(width: int, height: int) -> str:
    """Returns the aspect ratio as a simplified W:H string, e.g. '16:9'."""
    g = math.gcd(width, height)
    return f"{width // g}:{height // g}"


# ---------------------------------------------------------------------------
# 4. Color Depth  (Slide: "Color Depth (Bit Depth)")
#    N_colors = 2^b
# ---------------------------------------------------------------------------
def colors_from_bit_depth(bits: int) -> int:
    return 2 ** bits


# ---------------------------------------------------------------------------
# 5. Frame Buffer Size  (Slide: "Worked Example: Frame Buffer Size")
# ---------------------------------------------------------------------------
def calculate_framebuffer_size(width: int, height: int, bpp: int) -> float:
    """Returns frame buffer size in MB for given width, height and bits-per-pixel."""
    total_pixels = width * height
    bytes_per_pixel = bpp / 8.0
    memory_bytes = total_pixels * bytes_per_pixel
    return memory_bytes / (1024 ** 2)  # Convert to MB


# ---------------------------------------------------------------------------
# 6. Binary Thresholding  (Slide: "Monochrome Systems & Thresholding")
#    B(x,y) = 1 if I(x,y) >= T else 0
# ---------------------------------------------------------------------------
def threshold_image(image: np.ndarray, T: int) -> np.ndarray:
    """Converts a grayscale image array (values 0-255) into a binary (0/1) array."""
    return np.where(image >= T, 1, 0).astype(np.uint8)


# ---------------------------------------------------------------------------
# 7. Pixel Channel Bitmasking  (Slide: "Lab Exercise 2: Pixel Channel Bitmasking")
# ---------------------------------------------------------------------------
def extract_channel(img_rgb: np.ndarray, channel: str) -> np.ndarray:
    """
    Extract a single channel ('R', 'G', or 'B') from an HxWx3 RGB image array,
    zeroing out the other two channels (matches the lab exercise pattern).
    """
    channel = channel.upper()
    idx_map = {"R": 0, "G": 1, "B": 2}
    if channel not in idx_map:
        raise ValueError("channel must be one of 'R', 'G', 'B'")

    mask = img_rgb.copy()
    for name, idx in idx_map.items():
        if name != channel:
            mask[:, :, idx] = 0
    return mask


# ---------------------------------------------------------------------------
# 8. Downsampling via Stride  (Slide: "Lab Focus & Implementation" - Stride S)
# ---------------------------------------------------------------------------
def downsample_stride(image: np.ndarray, stride: int) -> np.ndarray:
    """Downsample a 2D or 3D array by taking every `stride`-th pixel in H and W."""
    return image[::stride, ::stride]


# ---------------------------------------------------------------------------
# 9. Video Throughput  (Slide: "Worked Example: Video Throughput")
# ---------------------------------------------------------------------------
def video_throughput_gbps(width: int, height: int, bpp: int, fps: int) -> float:
    frame_bits = width * height * bpp
    bitrate_bps = frame_bits * fps
    return bitrate_bps / 1e9  # Convert to Gbps


# ---------------------------------------------------------------------------
# 10. Compression Ratio synthesis  (Slide: "Check Your Understanding: Synthesis")
# ---------------------------------------------------------------------------
def required_compression_ratio(width: int, height: int, bpp: int, fps: int,
                                network_mbps: float):
    """
    Returns (uncompressed_mbps, is_possible, required_ratio)
    required_ratio is None if transmission is already possible.
    """
    uncompressed_gbps = video_throughput_gbps(width, height, bpp, fps)
    uncompressed_mbps = uncompressed_gbps * 1000  # Gbps -> Mbps

    if uncompressed_mbps <= network_mbps:
        return uncompressed_mbps, True, None

    ratio = uncompressed_mbps / network_mbps
    return uncompressed_mbps, False, ratio


# ===========================================================================
# DEMONSTRATION / SELF-TEST
# Reproduces every worked example from the slides and checks the numbers.
# ===========================================================================
def run_all_demos():
    print("=" * 70)
    print("1. TOTAL PIXEL COUNT  (Slide 4)")
    print("=" * 70)
    total = total_pixel_count(1920, 1080)
    print(f"1920 x 1080 = {total:,} pixels ({total / 1_000_000:.2f} MP)")
    assert total == 2_073_600

    print("\n" + "=" * 70)
    print("2. PIXELS PER INCH (PPI)  (Slide 6 worked example)")
    print("=" * 70)
    ppi = calculate_ppi(1170, 2532, 6.1)
    print(f"6.1-inch display, 1170x2532 -> PPI = {ppi:.2f}")
    assert round(ppi, 2) == 457.25

    print("\n" + "=" * 70)
    print("3. ASPECT RATIO  (Slide 7-8)")
    print("=" * 70)
    ar = aspect_ratio(1920, 1080)
    print(f"1920x1080 aspect ratio = {ar:.4f} ({simplify_aspect_ratio(1920, 1080)})")
    ar_43 = aspect_ratio(4, 3)
    print(f"4:3 aspect ratio = {ar_43:.4f}")
    assert simplify_aspect_ratio(1920, 1080) == "16:9"

    print("\n" + "=" * 70)
    print("4. COLOR DEPTH  (Slide 10)")
    print("=" * 70)
    for b in [1, 8, 24]:
        print(f"{b}-bit -> {colors_from_bit_depth(b):,} colors")
    assert colors_from_bit_depth(1) == 2
    assert colors_from_bit_depth(8) == 256
    assert colors_from_bit_depth(24) == 16_777_216

    print("\n" + "=" * 70)
    print("5. FRAME BUFFER SIZE  (Slide 13 worked example: 4K UHD, 32-bit RGBA)")
    print("=" * 70)
    mb_4k = calculate_framebuffer_size(3840, 2160, 32)
    print(f"4K UHD (3840x2160), 32-bit RGBA -> {mb_4k:.2f} MB")
    assert round(mb_4k, 2) == 31.64

    mb_1080_24 = calculate_framebuffer_size(1920, 1080, 24)
    print(f"1080p (1920x1080), 24-bit RGB   -> {mb_1080_24:.2f} MB")

    print("\n" + "=" * 70)
    print("6. BINARY THRESHOLDING  (Slide 12)")
    print("=" * 70)
    gray = np.array([[50, 120, 200],
                      [10, 128, 255]], dtype=np.uint8)
    binary = threshold_image(gray, T=128)
    print("Grayscale input:\n", gray)
    print("Threshold T=128 -> Binary output:\n", binary)
    expected = np.array([[0, 0, 1], [0, 1, 1]])
    assert np.array_equal(binary, expected)

    print("\n" + "=" * 70)
    print("7. PIXEL CHANNEL BITMASKING  (Slide 20, Lab Exercise 2)")
    print("=" * 70)
    img_rgb = np.array([[[255, 0, 0], [0, 255, 0]],
                         [[0, 0, 255], [255, 255, 0]]], dtype=np.uint8)
    red_only = extract_channel(img_rgb, "R")
    print("Original RGB image:\n", img_rgb)
    print("Red channel only:\n", red_only)
    expected_red = np.array([[[255, 0, 0], [0, 0, 0]],
                              [[0, 0, 0], [255, 0, 0]]], dtype=np.uint8)
    assert np.array_equal(red_only, expected_red)

    print("\n" + "=" * 70)
    print("8. DOWNSAMPLING (STRIDE)  (Slide 18, Lab Data Flow)")
    print("=" * 70)
    big_array = np.arange(64).reshape(8, 8)
    small_array = downsample_stride(big_array, stride=2)
    print("Original 8x8 array:\n", big_array)
    print("Downsampled (stride=2) ->\n", small_array)
    assert small_array.shape == (4, 4)

    print("\n" + "=" * 70)
    print("9. VIDEO THROUGHPUT  (Slide 15 worked example: 1080p, 24-bit, 60 FPS)")
    print("=" * 70)
    throughput = video_throughput_gbps(1920, 1080, 24, 60)
    print(f"1080p, 24-bit, 60 FPS -> {throughput:.2f} Gbps")
    assert round(throughput, 2) == 2.99

    print("\n" + "=" * 70)
    print("10. COMPRESSION RATIO SYNTHESIS  (Slide 21: 100 Mbps network, 30 FPS)")
    print("=" * 70)
    uncompressed_mbps, possible, ratio = required_compression_ratio(
        1920, 1080, 24, 30, network_mbps=100
    )
    print(f"Uncompressed bitrate at 30 FPS = {uncompressed_mbps:.0f} Mbps")
    print(f"Fits in 100 Mbps network? {possible}")
    if not possible:
        print(f"Required compression ratio ≈ {ratio:.1f} : 1")
    # Note: slide rounds the 60fps throughput (2.99 Gbps) then halves it for 30fps,
    # giving ~1490 Mbps. Computing exactly from raw pixel/bit counts gives 1493 Mbps.
    # Both round to the same ~14.9:1 compression ratio conclusion.
    assert round(uncompressed_mbps) == 1493
    assert possible is False
    assert round(ratio, 1) == 14.9

    print("\n" + "=" * 70)
    print("ALL DEMOS RAN SUCCESSFULLY - ALL ASSERTIONS PASSED ✅")
    print("=" * 70)


if __name__ == "__main__":
    run_all_demos()
