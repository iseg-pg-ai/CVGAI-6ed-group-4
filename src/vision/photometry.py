"""Photometric illumination correction, contrast enhancement, and baseline deskew.

Ref: Docs/implementation_plan.md, section 7, WP-04.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.vision.geometry import compute_paper_border_value


def estimate_skew_angle(image: np.ndarray, max_skew_deg: float = 25.0) -> float:
    """Estimate dominant text-line skew angle in degrees using OpenCV morphology and Hough lines.

    Args:
        image: BGR or grayscale receipt image.
        max_skew_deg: Maximum absolute skew angle in degrees to consider.

    Returns:
        Estimated skew angle in degrees in [-max_skew_deg, max_skew_deg], or 0.0 when no text
        lines are detected.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    h, w = gray.shape[:2]
    if h < 32 or w < 32:
        return 0.0

    bh_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, bh_kernel)
    if int(blackhat.max()) < 15:
        return 0.0

    _, binary = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    strip_w = max(15, w // 25)
    connect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (strip_w, 1))
    connected = cv2.dilate(binary, connect_kernel, iterations=1)

    lines = cv2.HoughLinesP(
        connected,
        rho=1.0,
        theta=np.pi / 180.0,
        threshold=40,
        minLineLength=max(30, w // 8),
        maxLineGap=max(10, w // 30),
    )
    if lines is None:
        return 0.0

    angles: list[float] = []
    for line in lines.reshape(-1, 4):
        x1, y1, x2, y2 = line
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        if np.hypot(dx, dy) < 1.0:
            continue
        angle = float(np.degrees(np.arctan2(dy, dx)))
        if -max_skew_deg <= angle <= max_skew_deg:
            angles.append(angle)

    if len(angles) < 3:
        return 0.0
    return float(np.median(angles))


def deskew_image(
    image: np.ndarray,
    min_angle_deg: float = 0.5,
    max_skew_deg: float = 25.0,
) -> tuple[np.ndarray, float]:
    """Deskew receipt image continuously using OpenCV affine rotation around text baselines.

    Args:
        image: Source image (BGR or grayscale).
        min_angle_deg: Minimum absolute angle required to trigger resampling.
        max_skew_deg: Maximum allowed skew angle in degrees.

    Returns:
        Tuple of (deskewed_image, applied_angle_deg).
    """
    angle = estimate_skew_angle(image, max_skew_deg=max_skew_deg)
    if abs(angle) < min_angle_deg:
        return image.copy(), 0.0

    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos_a = abs(float(rot_mat[0, 0]))
    sin_a = abs(float(rot_mat[0, 1]))
    new_w = max(1, int(round((h * sin_a) + (w * cos_a))))
    new_h = max(1, int(round((h * cos_a) + (w * sin_a))))
    rot_mat[0, 2] += (new_w / 2.0) - center[0]
    rot_mat[1, 2] += (new_h / 2.0) - center[1]

    border_val = compute_paper_border_value(image)
    deskewed = cv2.warpAffine(
        image,
        rot_mat,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_val,  # type: ignore[arg-type]
    )
    return deskewed, angle


def attenuate_shadows(gray: np.ndarray, kernel_size: int = 25) -> np.ndarray:
    """Normalize background lighting on crumpled paper via morphological division.

    Args:
        gray: Single-channel grayscale image.
        kernel_size: Size of the square morphological structuring element.

    Returns:
        Shadow-attenuated grayscale image.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    blur_ksize = kernel_size | 1
    background = cv2.GaussianBlur(background, (blur_ksize, blur_ksize), 0)
    background = np.maximum(background, 1)
    return cv2.divide(gray, background, scale=255)


def enhance_contrast_clahe(
    gray: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)
) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization to recover thermal print.

    Args:
        gray: Single-channel grayscale image.
        clip_limit: Threshold for contrast limiting. Values <= 0 return input untouched.
        tile_grid_size: Size of grid for histogram equalization.

    Returns:
        Enhanced grayscale image.
    """
    if clip_limit <= 0:
        return gray.copy()
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)
