"""Calculate normalized CER and WER for an OCR sample against its reference.

Example: python calculate_error.py sample_1
"""

import argparse
from pathlib import Path

from mark import error_rates


PROJECT_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample", nargs="?", default="sample_1", help="sample name or image path")
    parser.add_argument("--output", type=Path, help="override the OCR output path")
    parser.add_argument("--reference", type=Path, help="override the reference path")
    args = parser.parse_args()

    name = Path(args.sample).stem
    output_path = args.output or PROJECT_DIR / "output" / name / "final.txt"
    reference_path = args.reference or PROJECT_DIR / "reference" / f"{name}_reference.txt"
    output = output_path.read_text(encoding="utf-8")
    reference = reference_path.read_text(encoding="utf-8")
    rates = error_rates(output, reference)

    print(f"Output: {output_path}")
    print(f"Reference: {reference_path}")
    print(f"CER: {rates['cer']:.2%} ({rates['char_edits']}/{rates['reference_characters']} characters)")
    print(f"WER: {rates['wer']:.2%} ({rates['word_edits']}/{rates['reference_words']} words)")


if __name__ == "__main__":
    main()
