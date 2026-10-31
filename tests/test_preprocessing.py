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
    enhance_contrast_clahe,
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

    main(argv=["--file", str(img_path), "--rotate", "180", "--no-osd"])

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
