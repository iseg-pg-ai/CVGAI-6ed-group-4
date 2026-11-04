"""Grid-search and coordinate-ascent hyperparameter tuner for receipt preprocessing.

Evaluates photometric parameter combinations (CLAHE clip limit, ink contrast gamma,
unsharp mask strength, and shadow kernel size) against all 27 receipts in the golden set.

Ref: Docs/implementation_plan.md.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from src.evaluation.golden_set import carregar_ground_truth
from src.evaluation.scoring import agregar, pontuar_registo
from src.extraction.regex import extract as extract_regex
from src.ocr.tesseract import TesseractEngine
from src.vision.preprocessing import PreprocessOptions, preprocess

if TYPE_CHECKING:
    from src.core.settings import Settings


def prepare_aligned_corpus(receipts_dir: Path, records: list) -> list[tuple[Any, np.ndarray, int]]:
    """Precompute and cache aligned cropped images for each receipt to speed up tuning."""
    print("Pre-aligning and cropping corpus for fast photometric tuning...")
    cached = []
    for record in records:
        img_path = receipts_dir / record.filename
        raw = cv2.imread(str(img_path))
        if raw is None:
            raise ValueError(f"Unable to read {img_path}")
        # Run geometric stage once (no CLAHE/shadows/gamma/sharpen)
        base_opts = PreprocessOptions(
            max_side=1280,
            rotation_deg="auto",
            auto_crop=True,
            remove_shadows=False,
            clahe_clip_limit=0.0,
            auto_orient_osd=True,
        )
        prep = preprocess(raw, base_opts)
        cached.append((record, prep.image, prep.rotation_deg))
    print(f"Cached {len(cached)} pre-aligned receipt images.")
    return cached


def evaluate_params(
    cached_corpus: list[tuple[Any, np.ndarray, int]],
    ocr_engine: Any,
    clahe: float,
    gamma: float,
    sharpen: float,
    shadow_k: int,
) -> dict[str, Any]:
    """Evaluate a single photometric parameter configuration across the cached corpus."""
    all_results = []
    for record, aligned_bgr, _ in cached_corpus:
        gray = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2GRAY)
        if shadow_k > 0:
            from src.vision.photometry import attenuate_shadows

            gray = attenuate_shadows(gray, kernel_size=shadow_k)
        if clahe > 0.0:
            from src.vision.photometry import enhance_contrast_clahe

            gray = enhance_contrast_clahe(gray, clip_limit=clahe)
        if gamma != 1.0:
            from src.vision.photometry import adjust_ink_contrast

            gray = adjust_ink_contrast(gray, gamma=gamma)
        if sharpen > 0.0:
            from src.vision.photometry import sharpen_unsharp_mask

            gray = sharpen_unsharp_mask(gray, strength=sharpen)

        processed_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        doc = ocr_engine.read(processed_bgr)
        extraction = extract_regex(doc)
        matches = pontuar_registo(record, extraction)
        all_results.extend(matches)

    agg = agregar(all_results)
    global_metrics = agg["global"]
    amarrotado_metrics = agg.get("por_grupo", {}).get("amarrotado", {"taxa_acerto": 0.0})
    return {
        "clahe": clahe,
        "gamma": gamma,
        "sharpen": sharpen,
        "shadow_k": shadow_k,
        "taxa_acerto": global_metrics["taxa_acerto"],
        "taxa_acerto_amarrotado": amarrotado_metrics["taxa_acerto"],
        "taxa_omissao": global_metrics["taxa_omissao"],
        "taxa_falso_positivo": global_metrics["taxa_falso_positivo"],
    }


def main(argv: Sequence[str] | None = None, settings_override: Settings | None = None) -> None:
    """Run grid search tuning over photometric preprocessing parameters."""
    from src.core.settings import get_settings

    settings = settings_override or get_settings()
    parser = argparse.ArgumentParser(description="Tune preprocessing photometric parameters")
    parser.add_argument("--csv", type=Path, default=settings.ground_truth_path, help="Ground truth CSV path")
    parser.add_argument("--receipts-dir", type=Path, default=settings.receipts_dir, help="Receipts directory")
    parser.add_argument("--mode", choices=["grid", "fast"], default="fast", help="Tuning search mode")
    effective_argv = list(argv) if argv is not None else (sys.argv[1:] if __name__ == "__main__" else [])
    args = parser.parse_args(effective_argv)

    records = carregar_ground_truth(args.csv, receipts_dir=args.receipts_dir)
    cached_corpus = prepare_aligned_corpus(args.receipts_dir, records)

    # Use Tesseract engine for fast, deterministic CPU-based evaluation during tuning
    print("Initializing Tesseract engine for fast evaluation...")
    ocr_engine = TesseractEngine(lang="por", psm=6, tesseract_cmd=settings.tesseract_cmd)

    # Define grids
    clahe_grid = [1.5, 2.0, 2.5, 3.0]
    gamma_grid = [1.0, 1.15, 1.3, 1.45]
    sharpen_grid = [0.0, 0.4, 0.8, 1.2]
    shadow_grid = [25]

    if args.mode == "fast":
        # Coordinate ascent: start from baseline (2.0, 1.0, 0.0, 25)
        best = evaluate_params(cached_corpus, ocr_engine, 2.0, 1.0, 0.0, 25)
        print(f"Baseline: {best}")

        # Optimize CLAHE & Gamma
        print("Optimizing CLAHE and Gamma...")
        for c in clahe_grid:
            for g in gamma_grid:
                res = evaluate_params(cached_corpus, ocr_engine, c, g, best["sharpen"], best["shadow_k"])
                if res["taxa_acerto"] > best["taxa_acerto"] or (
                    res["taxa_acerto"] == best["taxa_acerto"]
                    and res["taxa_acerto_amarrotado"] > best["taxa_acerto_amarrotado"]
                ):
                    best = res
                    print(f"New best: {best}")

        # Optimize Sharpen
        print("Optimizing Sharpen strength...")
        for s in sharpen_grid:
            res = evaluate_params(cached_corpus, ocr_engine, best["clahe"], best["gamma"], s, best["shadow_k"])
            if res["taxa_acerto"] > best["taxa_acerto"] or (
                res["taxa_acerto"] == best["taxa_acerto"]
                and res["taxa_acerto_amarrotado"] > best["taxa_acerto_amarrotado"]
            ):
                best = res
                print(f"New best: {best}")

        best_params = best
    else:
        print("Running full grid search...")
        best_params = evaluate_params(
            cached_corpus, ocr_engine, clahe_grid[0], gamma_grid[0], sharpen_grid[0], shadow_grid[0]
        )
        best_score = best_params["taxa_acerto"]
        for c in clahe_grid:
            for g in gamma_grid:
                for s in sharpen_grid:
                    for sk in shadow_grid:
                        res = evaluate_params(cached_corpus, ocr_engine, c, g, s, sk)
                        print(
                            f"Tested C={c}, G={g}, S={s}, SK={sk} -> Acc: {res['taxa_acerto']:.4f} (Amarrotado: {res['taxa_acerto_amarrotado']:.4f})"
                        )
                        if res["taxa_acerto"] > best_score or (
                            res["taxa_acerto"] == best_score
                            and res["taxa_acerto_amarrotado"] > best_params["taxa_acerto_amarrotado"]
                        ):
                            best_score = res["taxa_acerto"]
                            best_params = res

    print("\n--- Tuning Winner ---")
    print(f"CLAHE clip limit: {best_params['clahe']}")
    print(f"Contrast gamma: {best_params['gamma']}")
    print(f"Sharpen strength: {best_params['sharpen']}")
    print(f"Shadow kernel size: {best_params['shadow_k']}")
    print(f"Global accuracy: {best_params['taxa_acerto']:.4f}")
    print(f"Crumpled group accuracy: {best_params['taxa_acerto_amarrotado']:.4f}")

    # Write tuning report
    out_dir = settings.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "tuning_report.md"
    report_path.write_text(
        f"# Tuning Report\n\n"
        f"- **CLAHE Clip Limit:** {best_params['clahe']}\n"
        f"- **Contrast Gamma:** {best_params['gamma']}\n"
        f"- **Sharpen Strength:** {best_params['sharpen']}\n"
        f"- **Shadow Kernel Size:** {best_params['shadow_k']}\n"
        f"- **Global Accuracy:** {best_params['taxa_acerto']:.4f}\n"
        f"- **Crumpled Group Accuracy:** {best_params['taxa_acerto_amarrotado']:.4f}\n",
        encoding="utf-8",
    )
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
