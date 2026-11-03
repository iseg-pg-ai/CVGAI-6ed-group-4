"""Unit tests for the OpenCV preprocessing pipeline."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.preprocessing import (
    PreprocessOptions,
    PreprocessResult,
    attenuate_shadows,
    crop_receipt,
    deskew_image,
    detect_upside_down_qr,
    enhance_contrast_clahe,
    estimate_skew_angle,
    order_corner_points,
    orient_image,
    preprocess,
    resize_max_side,
)


def test_resize_max_side_scales_down_large_image() -> None:
    """An image exceeding max_side is proportionally downscaled."""
    large = np.zeros((2400, 1200, 3), dtype=np.uint8)
    resized = resize_max_side(large, max_side=1280)

    assert resized.shape == (1280, 640, 3)


def test_resize_max_side_preserves_smaller_image() -> None:
    """An image smaller than max_side is copied without dimension changes."""
    small = np.zeros((600, 400, 3), dtype=np.uint8)
    resized = resize_max_side(small, max_side=1280)

    assert resized.shape == (600, 400, 3)


def test_orient_image_explicit_angles() -> None:
    """Explicit rotation options apply the correct geometric transform."""
    img = np.zeros((100, 200, 3), dtype=np.uint8)

    rot0, deg0 = orient_image(img, 0)
    assert rot0.shape == (100, 200, 3) and deg0 == 0

    rot90, deg90 = orient_image(img, 90)
    assert rot90.shape == (200, 100, 3) and deg90 == 90

    rot180, deg180 = orient_image(img, 180)
    assert rot180.shape == (100, 200, 3) and deg180 == 180

    rot270, deg270 = orient_image(img, 270)
    assert rot270.shape == (200, 100, 3) and deg270 == 270


def test_orient_image_auto_detects_landscape() -> None:
    """In auto mode, sideways receipts (width > height) are rotated clockwise."""
    landscape = np.zeros((300, 600, 3), dtype=np.uint8)
    portrait = np.zeros((600, 300, 3), dtype=np.uint8)

    rot_land, deg_land = orient_image(landscape, "auto")
    assert rot_land.shape == (600, 300, 3) and deg_land == 90

    rot_port, deg_port = orient_image(portrait, "auto")
    assert rot_port.shape == (600, 300, 3) and deg_port == 0


def test_order_corner_points_sorts_consistently() -> None:
    """Corner points are consistently ordered [top-left, top-right, bottom-right, bottom-left]."""
    scrambled = np.array([[200, 400], [0, 0], [0, 400], [200, 0]], dtype=np.float32)
    ordered = order_corner_points(scrambled)

    np.testing.assert_allclose(
        ordered,
        np.array([[0, 0], [200, 0], [200, 400], [0, 400]], dtype=np.float32),
    )


def test_crop_receipt_extracts_distinct_quadrilateral() -> None:
    """A prominent white rectangle on a dark canvas is detected and perspective-warped."""
    canvas = np.zeros((800, 600, 3), dtype=np.uint8)
    # Draw a prominent white rectangle occupying ~40% of the canvas
    cv2.rectangle(canvas, (100, 100), (500, 700), (255, 255, 255), thickness=-1)

    cropped = crop_receipt(canvas, min_area_ratio=0.20)

    # Should be cropped down close to the rectangle dimensions (400x600)
    h, w = cropped.shape[:2]
    assert 380 <= w <= 420
    assert 580 <= h <= 620


def test_crop_receipt_fallback_on_uniform_image() -> None:
    """When no distinct receipt contour exists, crop_receipt returns the input safely."""
    uniform = np.full((300, 200, 3), 128, dtype=np.uint8)
    result = crop_receipt(uniform)

    assert result.shape == uniform.shape
    np.testing.assert_array_equal(result, uniform)


def test_attenuate_shadows_flattens_lighting_gradient() -> None:
    """Morphological background division flattens artificial uneven lighting."""
    # Create an artificial horizontal lighting gradient from 40 to 220
    ramp = np.linspace(40, 220, 200, dtype=np.float32)
    gradient = np.tile(ramp, (200, 1)).astype(np.uint8)

    flattened = attenuate_shadows(gradient, kernel_size=25)

    # Standard deviation of pixel intensities across the gradient should decrease significantly
    assert np.std(flattened) < np.std(gradient)


def test_enhance_contrast_clahe_increases_contrast() -> None:
    """CLAHE increases contrast on a low-contrast synthetic image."""
    low_contrast = np.full((100, 100), 128, dtype=np.uint8)
    # Add subtle pattern
    low_contrast[25:75, 25:75] = 135

    enhanced = enhance_contrast_clahe(low_contrast, clip_limit=2.0)

    assert np.std(enhanced) > np.std(low_contrast)


def test_enhance_contrast_clahe_disabled_when_zero() -> None:
    """Zero or negative clip_limit leaves the image unchanged."""
    img = np.full((50, 50), 100, dtype=np.uint8)
    result = enhance_contrast_clahe(img, clip_limit=0.0)

    np.testing.assert_array_equal(result, img)


def test_preprocess_pipeline_contract() -> None:
    """The full preprocess pipeline returns PreprocessResult with all documented steps."""
    synthetic = np.full((400, 600, 3), 200, dtype=np.uint8)
    cv2.rectangle(synthetic, (150, 50), (450, 350), (255, 255, 255), thickness=-1)

    result = preprocess(synthetic, PreprocessOptions(max_side=500, rotation_deg="auto"))

    assert isinstance(result, PreprocessResult)
    assert result.image.ndim == 3 and result.image.shape[2] == 3
    assert result.image.dtype == np.uint8
    assert set(result.steps.keys()) == {"bruta", "rodada", "recortada", "filtrada"}
    for step_name, step_img in result.steps.items():
        assert step_img.ndim == 3 and step_img.shape[2] == 3, f"{step_name} is not 3-channel BGR"
        assert step_img.dtype == np.uint8


def test_preprocess_empty_image_raises_value_error() -> None:
    """Empty or zero-sized image raises ValueError."""
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="empty or zero-sized"):
        preprocess(empty)


def test_main_handles_missing_or_empty_receipts_dir(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """main prints a friendly message when receipts directory is missing or empty."""
    from src.preprocessing import main
    from src.settings import load_settings

    # Missing directory case
    settings = load_settings(repo_root=tmp_path)
    main(settings_override=settings)
    captured = capsys.readouterr().out
    assert "Receipts directory not found" in captured

    # Empty directory case
    settings.receipts_dir.mkdir(parents=True, exist_ok=True)
    main(settings_override=settings)
    captured = capsys.readouterr().out
    assert "No receipt images found" in captured


def test_main_executes_on_receipt_image(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """main loads the first image, preprocesses it, and triggers the visual display fallback."""
    from PIL import Image

    from src.preprocessing import main
    from src.settings import load_settings

    receipts_dir = tmp_path / "data" / "receipts"
    receipts_dir.mkdir(parents=True, exist_ok=True)
    img_path = receipts_dir / "test_receipt.jpg"
    synthetic = np.full((100, 100, 3), 200, dtype=np.uint8)
    cv2.imwrite(str(img_path), synthetic)

    shown_titles: list[str] = []

    def fake_show(self: Image.Image, title: str | None = None) -> None:
        if title:
            shown_titles.append(title)

    monkeypatch.setattr(Image.Image, "show", fake_show)

    settings = load_settings(repo_root=tmp_path)
    main(settings_override=settings)

    assert len(shown_titles) == 1
    assert "Antes vs Depois" in shown_titles[0]


def test_detect_orientation_osd_handles_missing_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    """detect_orientation_osd returns 0 safely when tesseract is not in PATH."""
    from src.preprocessing import detect_orientation_osd

    monkeypatch.setattr("shutil.which", lambda _name: None)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    assert detect_orientation_osd(img) == 0


def test_detect_orientation_osd_parses_rotation(monkeypatch: pytest.MonkeyPatch) -> None:
    """detect_orientation_osd parses Rotate: 180 when confidence exceeds threshold."""
    import subprocess

    from src.preprocessing import detect_orientation_osd

    mock_stdout = b"Rotate: 180\nOrientation confidence: 4.5\nScript: Latin\n"

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=mock_stdout)

    monkeypatch.setattr("shutil.which", lambda _name: "/mock/tesseract")
    monkeypatch.setattr("subprocess.run", fake_run)

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    assert detect_orientation_osd(img) == 180


def test_detect_orientation_osd_ignores_low_confidence(monkeypatch: pytest.MonkeyPatch) -> None:
    """detect_orientation_osd returns 0 when confidence is below min_confidence."""
    import subprocess

    from src.preprocessing import detect_orientation_osd

    mock_stdout = b"Rotate: 180\nOrientation confidence: 0.8\nScript: Latin\n"

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=mock_stdout)

    monkeypatch.setattr("shutil.which", lambda _name: "/mock/tesseract")
    monkeypatch.setattr("subprocess.run", fake_run)

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    assert detect_orientation_osd(img, min_confidence=2.0) == 0


def test_main_cli_file_and_rotate_flags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """main honors explicit --file and --rotate CLI arguments."""
    from PIL import Image

    from src.preprocessing import main

    img_path = tmp_path / "custom_receipt.jpg"
    synthetic = np.full((120, 80, 3), 180, dtype=np.uint8)
    cv2.imwrite(str(img_path), synthetic)

    shown_titles: list[str] = []

    def fake_show(self: Image.Image, title: str | None = None) -> None:
        if title:
            shown_titles.append(title)

    monkeypatch.setattr(Image.Image, "show", fake_show)

    main(argv=["--file", str(img_path), "--rotate", "180", "--no-osd", "--no-deskew"])

    assert len(shown_titles) == 1
    assert "custom_receipt.jpg" in shown_titles[0]


def test_main_cli_save_flag(tmp_path: Path) -> None:
    """main with --save writes side-by-side comparison image to disk."""
    from src.preprocessing import main

    img_path = tmp_path / "receipt_to_save.jpg"
    out_path = tmp_path / "comparison.png"
    synthetic = np.full((120, 80, 3), 180, dtype=np.uint8)
    cv2.imwrite(str(img_path), synthetic)

    main(argv=["--file", str(img_path), "--save", str(out_path), "--no-osd"])

    assert out_path.is_file()
    saved = cv2.imread(str(out_path))
    assert saved is not None
    assert saved.shape[0] == 1000


def test_order_corner_points_handles_steeply_tilted_quad() -> None:
    """Tilted rectangles past 45 degrees produce 4 distinct, cyclically ordered corners."""
    rect = ((500.0, 500.0), (200.0, 600.0), 45.0)
    box = cv2.boxPoints(rect)
    ordered = order_corner_points(box)

    assert ordered.shape == (4, 2)
    # Check that all 4 points are pairwise distinct
    for i in range(4):
        for j in range(i + 1, 4):
            dist = np.linalg.norm(ordered[i] - ordered[j])
            assert dist > 10.0, f"Vertices {i} and {j} are too close (collapsed quad)"

    # Verify cyclic sides: (0, 1) and (2, 3) must be equal, (1, 2) and (3, 0) must be equal
    w1 = np.linalg.norm(ordered[1] - ordered[0])
    w2 = np.linalg.norm(ordered[3] - ordered[2])
    h1 = np.linalg.norm(ordered[2] - ordered[1])
    h2 = np.linalg.norm(ordered[0] - ordered[3])
    np.testing.assert_allclose(w1, w2, atol=1e-3)
    np.testing.assert_allclose(h1, h2, atol=1e-3)


def test_crop_receipt_straightens_diagonal_receipt_on_light_background() -> None:
    """A diagonal receipt ribbon on a beige canvas is detected and perspective-warped."""
    canvas = np.full((1000, 1200, 3), 130, dtype=np.uint8)  # Beige table
    # Draw tilted white receipt (250x700 at -25 degrees)
    rect = ((600.0, 500.0), (250.0, 700.0), -25.0)
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.fillPoly(canvas, [box], (245, 245, 245))

    cropped = crop_receipt(canvas, min_area_ratio=0.10)

    # Output must be straightened into a portrait image close to (700, 250)
    h, w = cropped.shape[:2]
    assert h > w
    assert 650 <= h <= 750
    assert 220 <= w <= 280


def test_crop_receipt_min_area_rect_fallback_on_non_4_vertex_contour() -> None:
    """Contour with serrated/clipped corner falls back to minAreaRect 4-point warp."""
    canvas = np.zeros((1000, 1000, 3), dtype=np.uint8)
    # Create a 5-point polygon (rectangle with one corner cut off)
    polygon = np.array(
        [[200, 200], [600, 200], [600, 700], [500, 800], [200, 800]],
        dtype=np.int32,
    )
    cv2.fillPoly(canvas, [polygon], (255, 255, 255))

    cropped = crop_receipt(canvas, min_area_ratio=0.15)

    h, w = cropped.shape[:2]
    assert h > w
    assert 550 <= h <= 650
    assert 350 <= w <= 450


def test_estimate_skew_angle_returns_zero_on_blank_image() -> None:
    """estimate_skew_angle safely returns 0.0 on blank or tiny images."""
    blank = np.full((100, 100, 3), 255, dtype=np.uint8)
    assert estimate_skew_angle(blank) == 0.0

    tiny = np.zeros((10, 10, 3), dtype=np.uint8)
    assert estimate_skew_angle(tiny) == 0.0


def test_deskew_image_corrects_tilted_text_lines() -> None:
    """deskew_image detects text line tilt and rotates baselines toward horizontal."""
    canvas = np.full((500, 400, 3), 240, dtype=np.uint8)
    for y in range(80, 420, 25):
        cv2.putText(canvas, "FATURA-RECIBO 2026/001 TOTAL 15.50", (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    # Rotate by -12 degrees
    center = (200.0, 250.0)
    rot_mat = cv2.getRotationMatrix2D(center, -12.0, 1.0)
    tilted = cv2.warpAffine(canvas, rot_mat, (400, 500), borderValue=(240, 240, 240))

    deskewed, applied_angle = deskew_image(tilted, min_angle_deg=0.5)

    assert abs(applied_angle - 12.0) < 3.0
    # Remaining skew on deskewed image should be negligible
    remaining_skew = estimate_skew_angle(deskewed)
    assert abs(remaining_skew) < 2.0


def test_detect_upside_down_qr_checks_vertical_position(monkeypatch: pytest.MonkeyPatch) -> None:
    """detect_upside_down_qr returns True when QR centroid is in the upper 45%."""
    portrait = np.zeros((800, 400, 3), dtype=np.uint8)

    # Case 1: QR code detected at top (y = 100) -> upside down
    top_points = np.array([[[150, 80], [250, 80], [250, 180], [150, 180]]], dtype=np.float32)

    class MockDetectorTop:
        def detect(self, _img: np.ndarray) -> tuple[bool, np.ndarray]:
            return True, top_points

    monkeypatch.setattr("cv2.QRCodeDetector", MockDetectorTop)
    assert detect_upside_down_qr(portrait) is True

    # Case 2: QR code detected at bottom (y = 650) -> right side up
    bottom_points = np.array([[[150, 600], [250, 600], [250, 700], [150, 700]]], dtype=np.float32)

    class MockDetectorBottom:
        def detect(self, _img: np.ndarray) -> tuple[bool, np.ndarray]:
            return True, bottom_points

    monkeypatch.setattr("cv2.QRCodeDetector", MockDetectorBottom)
    assert detect_upside_down_qr(portrait) is False


def test_crop_receipt_ignores_off_center_textured_background() -> None:
    """crop_receipt keeps the centered neutral white receipt and ignores saturated wood-grain borders."""
    # Warm saturated wood-like background (BGR: low blue, medium green, high red)
    canvas = np.full((800, 600, 3), (60, 120, 175), dtype=np.uint8)
    # Add dense wood-grain stripes on the right side (x = 420..600)
    for y in range(0, 800, 6):
        cv2.line(canvas, (420, y), (599, min(799, y + 30)), (40, 90, 140), 2)

    # Centered neutral white receipt (x = 120..420, y = 40..760)
    cv2.rectangle(canvas, (120, 40), (420, 760), (240, 240, 240), thickness=-1)

    cropped = crop_receipt(canvas, min_area_ratio=0.15)
    h, w = cropped.shape[:2]

    # Should isolate the 300x720 receipt without picking the striped wood margin
    assert 680 <= h <= 760
    assert 280 <= w <= 330
    # Center of cropped image must be bright white paper
    assert float(cropped[h // 2, w // 2].mean()) > 220.0
