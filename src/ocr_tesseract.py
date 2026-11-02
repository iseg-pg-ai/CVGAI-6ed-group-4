"""Implementação do motor OCR partilhado no Tesseract.

Os resultados do Tesseract ao nível da palavra são agrupados em linhas de recibo
com geometria delimitada e identificadores baseados num, para que os consumidores
possam utilizar o mesmo contrato que o backend do EasyOCR.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import pytesseract

from src.types import IMAGE_EXTENSIONS, BBox, OcrDocument, OcrLine

if TYPE_CHECKING:
    from src.settings import Settings

_TESSERACT_LOCK = threading.Lock()


class TesseractEngine:
    """Motor OCR baseado em Tesseract, que agrega caixas de palavras em linhas."""

    name = "tesseract"

    def __init__(
        self,
        lang: str = "por",
        psm: int = 6,
        tesseract_cmd: Path | str | None = None,
    ) -> None:
        """Configura o idioma do Tesseract, o modo de segmentação de páginas e o executável (opcional).

        Argumentos:

        lang: Código do idioma do Tesseract, sendo o padrão o português.

        psm: Modo de segmentação de páginas do Tesseract.

        tesseract_cmd: Caminho opcional para o executável do Tesseract.

        Exceções:

        ValueError: Se o idioma estiver vazio ou o modo de segmentação de páginas não for compatível.
        """
        if not lang.strip():
            raise ValueError("A linguagem Tesseract não pode estar vazia.")
        if psm not in range(14):
            raise ValueError("O valor de Tesseract PSM deve estar entre 0 e 13.")

        if tesseract_cmd is None:
            from src.settings import get_settings

            tesseract_cmd = get_settings().tesseract_cmd

        self.lang = lang
        self.psm = psm
        self.tesseract_cmd = str(tesseract_cmd) if tesseract_cmd is not None else None

    def read(self, image: np.ndarray) -> OcrDocument:
        """Reconhece texto e agrega os resultados de palavras do Tesseract em linhas.

        Argumentos:

        imagem: Imagem BGR ou em tons de cinzento, geralmente produzida por pré-processamento.

        Retorno:

        Um documento OCR com as linhas ordenadas por ordem de leitura.

        Exceções:

        ValueError: Se a imagem estiver vazia ou tiver um formato não suportado.

        pytesseract.TesseractNotFoundError: Se o executável do Tesseract não estiver disponível.

        pytesseract.TesseractError: Se o Tesseract não conseguir processar a imagem.
        """
        if image is None or image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
            raise ValueError("A imagem de entrada não pode estar vazia nem ter tamanho zero.")
        if image.ndim == 2:
            image_for_tesseract = image
        elif image.ndim == 3 and image.shape[2] == 3:
            image_for_tesseract = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("A imagem de entrada deve ser em tons de cinzento ou ter três canais BGR.")

        height, width = image.shape[:2]
        start_time = time.perf_counter()
        data = self._image_to_data(image_for_tesseract)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        required_fields = (
            "text",
            "conf",
            "left",
            "top",
            "width",
            "height",
            "block_num",
            "par_num",
            "line_num",
        )
        if any(field not in data for field in required_fields):
            raise ValueError("A saída do Tesseract não contém os dados de palavras necessários.")

        field_lengths = {len(data[field]) for field in required_fields}
        if len(field_lengths) != 1:
            raise ValueError("Tesseract output fields have inconsistent lengths.")

        grouped_words: dict[tuple[int, int, int], list[tuple[str, float, int, int, int, int]]] = {}
        for index, raw_text in enumerate(data["text"]):
            text = str(raw_text).strip()
            if not text:
                continue

            try:
                confidence = float(data["conf"][index])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Tesseract devolveu uma confiança inválida na palavra {index}.") from error
            if confidence < 0:
                continue

            try:
                left = int(data["left"][index])
                top = int(data["top"][index])
                word_width = int(data["width"][index])
                word_height = int(data["height"][index])
                block_num = int(data["block_num"][index])
                par_num = int(data["par_num"][index])
                line_num = int(data["line_num"][index])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Tesseract devolveu uma geometria inválida na palavra {index}.") from error

            x1 = max(0, min(width, left))
            y1 = max(0, min(height, top))
            x2 = max(x1, min(width, left + word_width))
            y2 = max(y1, min(height, top + word_height))
            if x2 <= x1 or y2 <= y1:
                continue

            key = (block_num, par_num, line_num)
            grouped_words.setdefault(key, []).append((text, confidence, x1, y1, x2, y2))

        grouped_lines: list[tuple[int, int, tuple[int, int, int], OcrLine]] = []
        for key, words in grouped_words.items():
            x1 = min(word[2] for word in words)
            y1 = min(word[3] for word in words)
            x2 = max(word[4] for word in words)
            y2 = max(word[5] for word in words)
            line_text = " ".join(word[0] for word in sorted(words, key=lambda word: (word[2], word[3])))
            confidence = sum(word[1] for word in words) / len(words) / 100.0
            confidence = max(0.0, min(1.0, confidence))
            bbox = BBox(x1=x1, y1=y1, x2=x2, y2=y2)
            grouped_lines.append((y1, x1, key, OcrLine(id=0, text=line_text, confidence=confidence, bbox=bbox)))

        grouped_lines.sort(key=lambda line: (line[0], line[1], line[2]))
        lines = [
            OcrLine(id=index + 1, text=line.text, confidence=line.confidence, bbox=line.bbox)
            for index, (_, _, _, line) in enumerate(grouped_lines)
        ]
        return OcrDocument(
            lines=lines,
            engine=self.name,
            elapsed_ms=elapsed_ms,
            image_size=(width, height),
        )

    def _image_to_data(self, image: np.ndarray) -> dict[str, list[Any]]:
        """Chama o pytesseract isolando a sua configuração executável em todo o processo."""
        with _TESSERACT_LOCK:
            tesseract_module = pytesseract.pytesseract
            original_cmd = tesseract_module.tesseract_cmd
            try:
                if self.tesseract_cmd is not None:
                    tesseract_module.tesseract_cmd = self.tesseract_cmd
                result = pytesseract.image_to_data(
                    image,
                    lang=self.lang,
                    config=f"--psm {self.psm}",
                    output_type=pytesseract.Output.DICT,
                )
            finally:
                tesseract_module.tesseract_cmd = original_cmd
        return result


def main(
    argv: Sequence[str] | None = None,
    settings_override: Settings | None = None,
) -> None:
    """Run Tesseract OCR on a receipt and print recognized numbered lines.

    Args:
        argv: Optional command-line arguments. Defaults to sys.argv[1:].
        settings_override: Optional custom Settings. When None, get_settings() is used.
    """
    from src.preprocessing import PreprocessOptions, preprocess
    from src.settings import get_settings

    parser = argparse.ArgumentParser(description="Tesseract document reading preview")
    parser.add_argument("--file", "-f", type=str, default=None, help="Specific receipt image path")
    parser.add_argument("--lang", type=str, default="por", help="Tesseract language code")
    parser.add_argument("--psm", type=int, default=6, help="Tesseract page segmentation mode (0-13)")
    parser.add_argument("--no-preprocess", action="store_true", help="Skip OpenCV preprocessing")

    effective_argv = list(argv) if argv is not None else (sys.argv[1:] if __name__ == "__main__" else [])
    args = parser.parse_args(effective_argv)

    settings = settings_override or get_settings()

    target_path: Path | None = None
    if args.file:
        candidate = Path(args.file)
        if not candidate.is_file():
            print(f"Error: Specified file not found: {candidate}")
            return
        target_path = candidate
    else:
        if not settings.receipts_dir.is_dir():
            print(f"Receipts directory not found: {settings.receipts_dir}")
            return
        receipt_files = sorted(
            [p for p in settings.receipts_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        )
        if not receipt_files:
            print(f"No receipt images found in {settings.receipts_dir}")
            return
        target_path = receipt_files[0]

    print(f"Loading receipt: {target_path.name}")
    raw_image = cv2.imread(str(target_path))
    if raw_image is None:
        print(f"Error: Unable to read image file: {target_path}")
        return

    if args.no_preprocess:
        image_to_read = raw_image
    else:
        print("Preprocessing receipt...")
        prep_result = preprocess(raw_image, PreprocessOptions())
        image_to_read = prep_result.image

    print("Initializing Tesseract engine...")
    engine = TesseractEngine(lang=args.lang, psm=args.psm, tesseract_cmd=settings.tesseract_cmd)

    print("Running OCR recognition...")
    doc = engine.read(image_to_read)

    print(f"OCR finished in {doc.elapsed_ms:.1f} ms. Lines found: {len(doc.lines)}\n")
    print("--- Extracted Text with Line Citations ---")
    print(doc.as_prompt_text())


if __name__ == "__main__":
    main()
