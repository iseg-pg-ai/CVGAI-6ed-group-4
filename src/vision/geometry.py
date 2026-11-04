"""Geometric contour detection, boundary fitting, and perspective transformation.

Ref: Docs/implementation_plan.md, section 7, WP-04.
"""

from __future__ import annotations

import cv2
import numpy as np


def resize_max_side(image: np.ndarray, max_side: int = 1280) -> np.ndarray:
    """Resize image proportionally if its longest dimension exceeds max_side.

    Args:
        image: Source BGR image.
        max_side: Maximum allowed width or height in pixels.

    Returns:
        Resized or copied image.
    """
    height, width = image.shape[:2]
    largest = max(height, width)
    if largest <= max_side:
        return image.copy()

    scale = max_side / largest
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def order_corner_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 (x, y) coordinates into [top-left, top-right, bottom-right, bottom-left].

    Uses clockwise angle sorting around the quadrilateral centroid so that strongly
    tilted rectangles never assign the same vertex to two corners.

    Args:
        pts: Array of shape (4, 2) containing point coordinates.

    Returns:
        Array of shape (4, 2) in float32 ordered [top-left, top-right, bottom-right, bottom-left].
    """
    pts_f = pts.reshape(4, 2).astype(np.float32)
    center = pts_f.mean(axis=0)

    angles = np.arctan2(pts_f[:, 1] - center[1], pts_f[:, 0] - center[0])
    cw_pts = pts_f[np.argsort(angles)]

    start_idx = int(np.argmin(cw_pts.sum(axis=1)))
    return np.roll(cw_pts, -start_idx, axis=0).astype(np.float32)


def compute_paper_border_value(image: np.ndarray, mask: np.ndarray | None = None) -> tuple[int, ...] | int:
    """Compute a neutral background fill color from paper pixels to avoid BORDER_REPLICATE streaks."""
    if image.ndim == 2:
        if mask is not None and np.any(mask > 0):
            return int(np.median(image[mask > 0]))
        return int(np.median(image))
    if mask is not None and np.any(mask > 0):
        med = np.median(image[mask > 0], axis=0)
    else:
        med = np.median(image.reshape(-1, image.shape[2]), axis=0)
    return tuple(int(round(float(x))) for x in med)


# Backward-compatible alias
_compute_paper_border_value = compute_paper_border_value


def warp_perspective_quad(
    image: np.ndarray,
    corners: np.ndarray,
    border_value: tuple[int, ...] | int = (240, 240, 240),
) -> np.ndarray:
    """Warp 4 corner points into a straightened portrait BGR image via perspective transform.

    Args:
        image: Source BGR image.
        corners: Array of 4 (x, y) corner coordinates.
        border_value: Constant fill color for any out-of-bounds border pixels.

    Returns:
        Perspective-warped upright image with height >= width.
    """
    rect = order_corner_points(corners)
    tl, tr, br, bl = rect

    width_a = float(np.linalg.norm(br - bl))
    width_b = float(np.linalg.norm(tr - tl))
    target_width = max(int(round(max(width_a, width_b))), 1)

    height_a = float(np.linalg.norm(tr - br))
    height_b = float(np.linalg.norm(tl - bl))
    target_height = max(int(round(max(height_a, height_b))), 1)

    dst = np.array(
        [
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1],
        ],
        dtype=np.float32,
    )

    transform_matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(
        image,
        transform_matrix,
        (target_width, target_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,  # type: ignore[arg-type]
    )

    warped_h, warped_w = warped.shape[:2]
    if warped_w > warped_h:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

    return warped


# Backward-compatible alias
_warp_perspective_quad = warp_perspective_quad


def crop_receipt_with_status(
    image: np.ndarray,
    min_area_ratio: float = 0.20,
    min_tilt_for_perspective_deg: float = 5.0,
) -> tuple[np.ndarray, bool]:
    """Adaptively segment receipt paper and either slice losslessly or warp perspective.

    Segments neutral bright paper (high HSV Value, low Saturation) covering the image center.
    When the receipt is nearly upright (tilt < min_tilt_for_perspective_deg), applies a lossless
    axis-aligned bounding-box crop so crumpled paper is neither sheared nor resampled. When
    genuinely angled (tilt >= min_tilt_for_perspective_deg), applies a padded 4-point perspective
    transform with constant paper-colored border fill.

    Returns:
        Tuple of (cropped_or_original_image, did_crop_boolean).
    """
    height, width = image.shape[:2]
    total_area = float(height * width)
    min_area = total_area * min_area_ratio
    max_area = total_area * 0.96

    if image.ndim == 3:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        v_chan = hsv[:, :, 2].astype(np.float32)
        s_chan = hsv[:, :, 1].astype(np.float32)
        paper_score = np.clip(v_chan - 1.2 * s_chan, 0, 255).astype(np.uint8)
    else:
        paper_score = image.copy()

    # Uniform image check (avoids false Otsu split on flat synthetic canvases)
    if int(paper_score.max()) - int(paper_score.min()) < 20:
        return image.copy(), False

    blurred = cv2.GaussianBlur(paper_score, (11, 11), 0)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    paper_closed = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, close_kernel)
    _, otsu_mask = cv2.threshold(paper_closed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(otsu_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image.copy(), False

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    center_pt = (width / 2.0, height / 2.0)
    min_center_dist = -0.10 * min(width, height)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            break
        if area > max_area:
            continue

        hull = cv2.convexHull(contour)
        if cv2.pointPolygonTest(hull, center_pt, True) < min_center_dist:
            continue

        rot_rect = cv2.minAreaRect(hull)
        (cx, cy), (rect_w, rect_h), ang = rot_rect
        rect_area = rect_w * rect_h
        if rect_area < min_area or rect_area > max_area:
            continue

        solidity = area / max(rect_area, 1.0)
        if solidity < 0.65:
            continue

        abs_ang = abs(ang) % 90.0
        tilt_deg = min(abs_ang, 90.0 - abs_ang)

        if tilt_deg >= min_tilt_for_perspective_deg:
            # Tilted receipt: pad rotated box outward by 6% so edge characters are never cut
            padded_rect = ((cx, cy), (rect_w * 1.06, rect_h * 1.06), ang)
            box_pts = cv2.boxPoints(padded_rect)
            border_val = compute_paper_border_value(image, otsu_mask)
            warped = warp_perspective_quad(image, box_pts, border_value=border_val)
            return warped, True

        # Upright receipt: lossless axis-aligned bounding-box crop
        bx, by, bw, bh = cv2.boundingRect(hull)
        pad_x = int(round(bw * 0.015))
        pad_y = int(round(bh * 0.015))
        x1 = max(0, bx - pad_x)
        y1 = max(0, by - pad_y)
        x2 = min(width, bx + bw + pad_x)
        y2 = min(height, by + bh + pad_y)
        cropped = image[y1:y2, x1:x2].copy()
        if cropped.shape[1] > cropped.shape[0]:
            cropped = cv2.rotate(cropped, cv2.ROTATE_90_CLOCKWISE)
        return cropped, True

    return image.copy(), False


# Backward-compatible alias
_crop_receipt_with_status = crop_receipt_with_status


def crop_receipt(image: np.ndarray, min_area_ratio: float = 0.20) -> np.ndarray:
    """Detect receipt boundary contour and warp perspective into an upright rectangle.

    Falls back cleanly to the input image when no distinct quadrilateral is found.

    Args:
        image: Source BGR image.
        min_area_ratio: Minimum ratio of receipt area relative to total image area.

    Returns:
        Warped rectangular receipt or uncropped image if detection fails.
    """
    warped, _ = crop_receipt_with_status(image, min_area_ratio=min_area_ratio)
    return warped
