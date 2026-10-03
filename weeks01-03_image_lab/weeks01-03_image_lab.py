# Name    :  Tooba
# Roll No : 2K24/AIE/66


import os

import cv2
import matplotlib

# "Agg" backend = draw into files only, never open a GUI window.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must come after matplotlib.use)
import numpy as np  # noqa: E402


# ---------------------------------------------------------------------------
# Small helper functions
# ---------------------------------------------------------------------------

def load_image_bgr(image_path):
    """Read an image with OpenCV. OpenCV always loads colour images as BGR."""
    if not os.path.isfile(image_path):
        raise FileNotFoundError("Image not found: " + str(image_path))
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not read image (bad or unsupported file): "
                         + str(image_path))
    return image


def make_output_dir(output_dir):
    """Create the output folder if it does not exist yet."""
    os.makedirs(output_dir, exist_ok=True)


def to_gray(image_bgr):
    """Convert a BGR image to a single-channel grayscale image (uint8)."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def bgr_to_rgb(image_bgr):
    """Matplotlib expects RGB, so swap channel order before showing."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def save_montage(panels, rows, cols, output_path, suptitle):
    """
    Save several labeled images in one figure.
    panels = list of (title, image) pairs.
    Gray images (2-D arrays) are shown in gray colormap with fixed 0-255 range
    so that brightness changes are really visible.
    Colour images must already be in RGB order.
    """
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.2 * rows))
    axes = np.array(axes).reshape(-1)

    for ax, (title, img) in zip(axes, panels):
        if img.ndim == 2:
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    # Hide any unused boxes in the grid
    for ax in axes[len(panels):]:
        ax.axis("off")

    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)

    if not os.path.isfile(output_path):
        raise IOError("Output image was not written: " + output_path)


# ---------------------------------------------------------------------------
# Task 1 - Image data
# ---------------------------------------------------------------------------

def inspect_image(image_path: str) -> dict:
    """Load the image and return its measured image-data properties."""
    image = load_image_bgr(image_path)

    height = int(image.shape[0])
    width = int(image.shape[1])
    channels = int(image.shape[2])
    pixel_count = width * height
    # 8 bits per channel = 1 byte per channel value
    estimated_bytes = pixel_count * channels * 1

    result = {
        "width": width,
        "height": height,
        "channels": channels,
        "shape": [int(v) for v in image.shape],
        "pixel_count": pixel_count,
        "estimated_bytes": estimated_bytes,
        # cv2.imread gives Blue-Green-Red order, not RGB
        "color_order": "BGR",
    }
    return result


# ---------------------------------------------------------------------------
# Task 2 - Colour and resolution
# ---------------------------------------------------------------------------

def split_channels(image_bgr):
    """
    Return three colour images (blue, green, red) that keep only one channel
    each. The other two channels are set to 0, so each image looks tinted.
    """
    blue_only = np.zeros_like(image_bgr)
    green_only = np.zeros_like(image_bgr)
    red_only = np.zeros_like(image_bgr)

    blue_only[:, :, 0] = image_bgr[:, :, 0]
    green_only[:, :, 1] = image_bgr[:, :, 1]
    red_only[:, :, 2] = image_bgr[:, :, 2]
    return blue_only, green_only, red_only


def downsample_half(image_bgr):
    """Make an image with half the width and half the height."""
    height, width = image_bgr.shape[:2]
    new_width = max(1, width // 2)
    new_height = max(1, height // 2)
    # INTER_AREA is the usual choice when shrinking an image
    return cv2.resize(image_bgr, (new_width, new_height),
                      interpolation=cv2.INTER_AREA)


def create_pixel_views(image_path: str, output_dir: str) -> dict:
    """Create the labeled channel, grayscale, and downsampled views."""
    image = load_image_bgr(image_path)
    make_output_dir(output_dir)

    height, width = image.shape[:2]
    blue_only, green_only, red_only = split_channels(image)
    gray = to_gray(image)
    small = downsample_half(image)
    small_h, small_w = small.shape[:2]

    panels = [
        ("Original (%d x %d)" % (width, height), bgr_to_rgb(image)),
        ("Red channel", bgr_to_rgb(red_only)),
        ("Green channel", bgr_to_rgb(green_only)),
        ("Blue channel", bgr_to_rgb(blue_only)),
        ("Grayscale", gray),
        ("Downsampled half (%d x %d)" % (small_w, small_h),
         bgr_to_rgb(small)),
    ]

    output_path = os.path.join(output_dir, "pixel_views.png")
    save_montage(panels, 2, 3, output_path,
                 "Pixel views: channels, grayscale, half resolution")

    return {
        "original_size": [int(width), int(height)],
        "downsampled_size": [int(small_w), int(small_h)],
        "output_path": output_path,
    }


# ---------------------------------------------------------------------------
# Task 3 - Adjustments
# ---------------------------------------------------------------------------

def adjust_brightness(gray, brightness_delta):
    """Add brightness_delta to every pixel and clip to 0..255."""
    # int16 so that e.g. 250 + 40 = 290 does not wrap around in uint8
    result = gray.astype(np.int16) + int(brightness_delta)
    result = np.clip(result, 0, 255)
    return result.astype(np.uint8)


def adjust_contrast(gray, contrast_factor):
    """Multiply every pixel by contrast_factor and clip to 0..255."""
    result = gray.astype(np.float32) * float(contrast_factor)
    result = np.clip(result, 0, 255)
    return result.astype(np.uint8)


def apply_threshold(gray, threshold):
    """Pixels greater than threshold become 255 (white), others 0 (black)."""
    result = np.where(gray > threshold, 255, 0)
    return result.astype(np.uint8)


def create_adjustments(
    image_path: str,
    output_dir: str,
    brightness_delta: int = 40,
    contrast_factor: float = 1.5,
    threshold: int = 127,
) -> dict:
    """Create labeled brightness, contrast, and threshold results."""
    # Validate inputs first
    if isinstance(threshold, bool) or not isinstance(threshold, (int, np.integer)):
        raise ValueError("threshold must be an integer between 0 and 255")
    if threshold < 0 or threshold > 255:
        raise ValueError("threshold must be in the range 0-255")

    image = load_image_bgr(image_path)
    make_output_dir(output_dir)
    gray = to_gray(image)

    bright = adjust_brightness(gray, brightness_delta)
    contrast = adjust_contrast(gray, contrast_factor)
    binary = apply_threshold(gray, threshold)

    panels = [
        ("Grayscale (original)", gray),
        ("Brighter (+%d)" % brightness_delta, bright),
        ("Higher contrast (x%s)" % contrast_factor, contrast),
        ("Threshold (> %d = white)" % threshold, binary),
    ]

    output_path = os.path.join(output_dir, "adjustments.png")
    save_montage(panels, 2, 2, output_path,
                 "Brightness, contrast and thresholding")

    return {
        "brightness_delta": brightness_delta,
        "contrast_factor": contrast_factor,
        "threshold": int(threshold),
        "output_path": output_path,
    }


# ---------------------------------------------------------------------------
# Task 4 - Blur and edges
# ---------------------------------------------------------------------------

def mean_blur(gray, kernel_size):
    """Mean (box) blur with a kernel_size x kernel_size window."""
    return cv2.blur(gray, (kernel_size, kernel_size))


def sobel_edges(gray):
    """
    Sobel edge strength.
    Sobel in x and y direction -> magnitude = sqrt(gx^2 + gy^2),
    then scaled to 0..255 so it can be shown as an image.
    """
    # CV_64F keeps negative gradient values (uint8 would cut them off)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

    max_value = magnitude.max()
    if max_value > 0:
        magnitude = magnitude / max_value * 255.0
    magnitude = np.clip(magnitude, 0, 255)
    return magnitude.astype(np.uint8)


def create_blur_and_edges(
    image_path: str,
    output_dir: str,
    kernel_size: int = 5,
) -> dict:
    """Create labeled grayscale, mean-blur, and Sobel-edge results."""
    # kernel_size must be a positive odd integer
    if isinstance(kernel_size, bool) or not isinstance(kernel_size, (int, np.integer)):
        raise ValueError("kernel_size must be a positive odd integer")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")

    image = load_image_bgr(image_path)
    make_output_dir(output_dir)

    gray = to_gray(image)
    blurred = mean_blur(gray, int(kernel_size))
    edges_original = sobel_edges(gray)
    edges_blurred = sobel_edges(blurred)

    panels = [
        ("Grayscale", gray),
        ("Mean blur (%dx%d)" % (kernel_size, kernel_size), blurred),
        ("Sobel edges - original gray", edges_original),
        ("Sobel edges - blurred gray", edges_blurred),
    ]

    output_path = os.path.join(output_dir, "blur_and_edges.png")
    save_montage(panels, 2, 2, output_path,
                 "Mean blur and Sobel edge detection")

    return {
        "kernel_size": int(kernel_size),
        "output_path": output_path,
    }


# ---------------------------------------------------------------------------
# Task 5 - Reproducible run
# ---------------------------------------------------------------------------

def run_lab(image_path: str, output_dir: str) -> dict:
    """Run Tasks 1-4 and return their results together."""
    results = {}
    results["task1"] = inspect_image(image_path)
    results["task2"] = create_pixel_views(image_path, output_dir)
    results["task3"] = create_adjustments(image_path, output_dir)
    results["task4"] = create_blur_and_edges(image_path, output_dir)
    return results


def main() -> None:
    """Run the lab using the required repository paths."""
    image_path = os.path.join("images", "original.jpg")
    output_dir = "outputs"

    os.makedirs(output_dir, exist_ok=True)
    results = run_lab(image_path, output_dir)

    # Print everything so the values can be copied into the README
    t1 = results["task1"]
    print("=== Task 1: Image data ===")
    print("Width x Height :", t1["width"], "x", t1["height"])
    print("Channels       :", t1["channels"])
    print("NumPy shape    :", t1["shape"])
    print("Pixel count    :", t1["pixel_count"])
    print("Estimated bytes:", t1["estimated_bytes"], "(8 bits per channel)")
    print("Color order    :", t1["color_order"])

    t2 = results["task2"]
    print("\n=== Task 2: Pixel views ===")
    print("Original size    [w, h]:", t2["original_size"])
    print("Downsampled size [w, h]:", t2["downsampled_size"])
    print("Saved:", t2["output_path"])

    t3 = results["task3"]
    print("\n=== Task 3: Adjustments ===")
    print("brightness_delta:", t3["brightness_delta"])
    print("contrast_factor :", t3["contrast_factor"])
    print("threshold       :", t3["threshold"])
    print("Saved:", t3["output_path"])

    t4 = results["task4"]
    print("\n=== Task 4: Blur and edges ===")
    print("kernel_size:", t4["kernel_size"])
    print("Saved:", t4["output_path"])

    print("\nDone. All three output images were created in", output_dir + "/")


if __name__ == "__main__":
    main()
