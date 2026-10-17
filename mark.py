"""Evaluate and improve an Arabic OCR transcript with prompt diversity.

Without a human reference, the reported WER is an image-grounded model estimate,
not a true measurement. Pass --reference to calculate exact WER and CER.

Example:
    python mark.py input/sample.jpg output/sample/final.txt
    python mark.py input/sample.jpg --reference sample_reference.txt
"""

import argparse
import json
import re
import unicodedata
from pathlib import Path

from PIL import Image

import main


MODEL_BLIND = main.MODEL_MARK_BLIND
MODEL_FORENSIC = main.MODEL_MARK_FORENSIC
MODEL_JUDGE = main.MODEL_MARK_JUDGE


def configure_models(blind: str, forensic: str, judge: str) -> None:
    """Use the same cost-quality profile as the calling OCR pipeline."""
    global MODEL_BLIND, MODEL_FORENSIC, MODEL_JUDGE
    MODEL_BLIND = blind
    MODEL_FORENSIC = forensic
    MODEL_JUDGE = judge


BLIND_PROMPT = """Act as a conservative Arabic paleographer. Produce a fresh,
independent transcription of this page without guessing from context.

Rules:
- Read every visible text line, including headings, dates, marginal text, and signatures.
- Preserve spelling, names, numerals, and line order exactly as seen.
- Read dates digit by digit and cross-check them visually before answering.
- Do not silently fix grammar or complete genealogical names from expectation.
- Use [غير واضح] only for the smallest unreadable span.
- Return ONLY a JSON array of strings, one string per visual line."""


FORENSIC_PROMPT = """You are performing forensic verification of handwritten
Arabic OCR. The current candidate is below. Inspect the image yourself and create
a complete corrected transcription.

Pay exceptional attention to:
1. Hijri and Gregorian dates, reading each digit independently.
2. Every personal, tribal, and place name.
3. ابن/بن, الشيخ, بريك, and similar short words where dots matter.
4. Numbered lists and whether their stated counts match their entries.
5. Missing, duplicated, or invented lines.

Do not prefer the candidate merely because it sounds plausible. Preserve an
uncertainty marker rather than inventing a name. Return ONLY a JSON array of
strings, one string per visual line.

CURRENT CANDIDATE:
{candidate}"""


JUDGE_PROMPT = """Act as the final visual adjudicator for Arabic handwriting.
The image is the only ground truth. Three OCR candidates are provided below.
They may share the same error, so do not decide by majority vote.

For every line, inspect the image and select, combine, or replace the candidates.
Verify dates digit by digit, preserve exact names and spelling, remove invented or
duplicated lines, and use [غير واضح] for the smallest genuinely unreadable span.

Return ONLY this JSON object:
{{
  "current_score_10": 0.0,
  "current_estimated_wer": 0.0,
  "final_score_10": 0.0,
  "final_estimated_wer": 0.0,
  "rationale": "short evidence-based assessment",
  "issues": ["remaining uncertainty 1"],
  "lines": ["final visual line 1", "final visual line 2"]
}}

The current fields grade Candidate A before correction; the final fields grade
your synthesized lines. Scores estimate fidelity from 0 to 10. WER values must
be fractions from 0 to 1 reflecting visible substitutions, deletions, and
insertions. They are estimates because no human reference is supplied.

CANDIDATE A — current pipeline:
{current}

CANDIDATE B — blind literal pass:
{blind}

CANDIDATE C — forensic pass:
{forensic}"""


def json_lines(lines: list[str]) -> str:
    return json.dumps(lines, ensure_ascii=False, indent=2)


def parse_lines(reply: str) -> list[str]:
    value = main.parse_json(reply)
    if not isinstance(value, list):
        raise ValueError("transcription response is not a JSON array")
    result = []
    for item in value:
        text = str(item.get("text", "") if isinstance(item, dict) else item).strip()
        if text:
            result.append(text)
    if not result:
        raise ValueError("transcription response contains no lines")
    return result


def normalize_for_metric(text: str) -> str:
    """Normalize presentation differences without changing Arabic letters."""
    text = unicodedata.normalize("NFKC", text).replace("\u0640", "")
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    text = re.sub(r"[^\w\u0600-\u06FF]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def edit_distance(reference, hypothesis) -> int:
    """Levenshtein distance for either token or character sequences."""
    previous = list(range(len(hypothesis) + 1))
    for i, reference_item in enumerate(reference, start=1):
        current = [i]
        for j, hypothesis_item in enumerate(hypothesis, start=1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (reference_item != hypothesis_item),
            ))
        previous = current
    return previous[-1]


def error_rates(candidate: str, reference: str) -> dict:
    """Calculate exact normalized WER/CER against a supplied reference."""
    candidate_normalized = normalize_for_metric(candidate)
    reference_normalized = normalize_for_metric(reference)
    candidate_words = candidate_normalized.split()
    reference_words = reference_normalized.split()
    candidate_chars = candidate_normalized.replace(" ", "")
    reference_chars = reference_normalized.replace(" ", "")
    word_edits = edit_distance(reference_words, candidate_words)
    char_edits = edit_distance(reference_chars, candidate_chars)
    return {
        "wer": word_edits / max(1, len(reference_words)),
        "cer": char_edits / max(1, len(reference_chars)),
        "word_edits": word_edits,
        "reference_words": len(reference_words),
        "char_edits": char_edits,
        "reference_characters": len(reference_chars),
    }


def pairwise_wer(left: list[str], right: list[str]) -> float:
    """Measure candidate disagreement; this is not accuracy or true WER."""
    return error_rates("\n".join(left), "\n".join(right))["wer"]


def transcribe(model: str, prompt: str, img: Image.Image) -> list[str]:
    return parse_lines(main.chat(model, prompt, [img]))


def adjudicate(
    img: Image.Image,
    current: list[str],
    blind: list[str],
    forensic: list[str],
) -> dict:
    prompt = JUDGE_PROMPT.format(
        current=json_lines(current),
        blind=json_lines(blind),
        forensic=json_lines(forensic),
    )
    result = main.parse_json(main.chat(MODEL_JUDGE, prompt, [img]))
    if not isinstance(result, dict) or not isinstance(result.get("lines"), list):
        raise ValueError("judge response does not contain a lines array")
    result["lines"] = [str(line).strip() for line in result["lines"] if str(line).strip()]
    if not result["lines"]:
        raise ValueError("judge returned an empty transcription")
    for field in ("current_score_10", "final_score_10"):
        try:
            result[field] = max(0.0, min(10.0, float(result.get(field, 0))))
        except (TypeError, ValueError):
            result[field] = 0.0
    for field in ("current_estimated_wer", "final_estimated_wer"):
        try:
            result[field] = max(0.0, min(1.0, float(result.get(field, 1))))
        except (TypeError, ValueError):
            result[field] = 1.0
    return result


def run(image_path: Path, candidate_path: Path, reference_path: Path | None = None) -> dict:
    img = Image.open(image_path).convert("RGB")
    current = candidate_path.read_text(encoding="utf-8").splitlines()
    current = [line.strip() for line in current if line.strip()]
    if not current:
        raise ValueError(f"candidate is empty: {candidate_path}")

    print(f"Blind pass: {MODEL_BLIND}")
    blind = transcribe(MODEL_BLIND, BLIND_PROMPT, img)
    print(f"Forensic pass: {MODEL_FORENSIC}")
    forensic = transcribe(
        MODEL_FORENSIC,
        FORENSIC_PROMPT.format(candidate=json_lines(current)),
        img,
    )
    print(f"Visual adjudication: {MODEL_JUDGE}")
    judgement = adjudicate(img, current, blind, forensic)
    final_lines = judgement["lines"]

    report = {
        "image": str(image_path),
        "candidate": str(candidate_path),
        "models": {
            "blind": MODEL_BLIND,
            "forensic": MODEL_FORENSIC,
            "judge": MODEL_JUDGE,
        },
        "model_profile": main.ACTIVE_MODEL_PROFILE,
        "current_score_10": judgement["current_score_10"],
        "current_estimated_wer": judgement["current_estimated_wer"],
        "final_score_10": judgement["final_score_10"],
        "final_estimated_wer": judgement["final_estimated_wer"],
        "estimated_wer_warning": "Model estimate only; use --reference for exact WER.",
        "rationale": str(judgement.get("rationale", "")),
        "issues": judgement.get("issues", []),
        "candidate_disagreement": {
            "current_vs_blind_wer": pairwise_wer(current, blind),
            "current_vs_forensic_wer": pairwise_wer(current, forensic),
            "blind_vs_forensic_wer": pairwise_wer(blind, forensic),
        },
        "line_counts": {
            "current": len(current),
            "blind": len(blind),
            "forensic": len(forensic),
            "final": len(final_lines),
        },
        "final_lines": final_lines,
    }

    if reference_path:
        reference = reference_path.read_text(encoding="utf-8")
        report["exact_metrics"] = {
            "current": error_rates("\n".join(current), reference),
            "blind": error_rates("\n".join(blind), reference),
            "forensic": error_rates("\n".join(forensic), reference),
            "final": error_rates("\n".join(final_lines), reference),
        }
    return report


def main_cli() -> None:
    parser = argparse.ArgumentParser(
        description="Retry, adjudicate, and evaluate an Arabic OCR transcript."
    )
    parser.add_argument("image", type=Path)
    parser.add_argument("candidate", type=Path, nargs="?")
    parser.add_argument("--reference", type=Path, help="human ground truth for exact WER/CER")
    parser.add_argument("--output", type=Path, help="path for improved transcription")
    parser.add_argument("--report", type=Path, help="path for JSON evaluation report")
    profile_group = parser.add_mutually_exclusive_group()
    for profile in ("free", "lite", "high", "max"):
        profile_group.add_argument(
            f"--{profile}",
            dest="model_profile",
            action="store_const",
            const=profile,
            help=f"use the {profile} model profile (default: lite)",
        )
    parser.set_defaults(model_profile="lite")
    args = parser.parse_args()

    main.configure_model_profile(args.model_profile)
    configure_models(
        main.MODEL_MARK_BLIND,
        main.MODEL_MARK_FORENSIC,
        main.MODEL_MARK_JUDGE,
    )

    image = main.resolve_input_path(args.image)
    artifact_name = main.sanitize_artifact_name(image.stem)
    log_dir, output_dir = main.artifact_directories(image)
    candidate = args.candidate or output_dir / "final.txt"
    output = args.output or log_dir / f"{artifact_name}_retry.txt"
    report_path = args.report or log_dir / f"{artifact_name}.mark.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report = run(image, candidate, args.reference)
    output.write_text("\n".join(report["final_lines"]), encoding="utf-8")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"Current estimate: {report['current_score_10']:.1f}/10, "
        f"WER {report['current_estimated_wer']:.1%}"
    )
    print(
        f"Retry estimate: {report['final_score_10']:.1f}/10, "
        f"WER {report['final_estimated_wer']:.1%} (not ground truth)"
    )
    if "exact_metrics" in report:
        print(f"Exact final WER: {report['exact_metrics']['final']['wer']:.1%}")
        print(f"Exact final CER: {report['exact_metrics']['final']['cer']:.1%}")
    print(f"Wrote {output}")
    print(f"Wrote {report_path}")
    print(
        f"API calls: {main.USAGE['calls']}, prompt tokens: {main.USAGE['prompt_tokens']}, "
        f"completion tokens: {main.USAGE['completion_tokens']}"
    )


if __name__ == "__main__":
    main_cli()
