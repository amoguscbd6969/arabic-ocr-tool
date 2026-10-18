"""Arabic translation pipeline with local and free candidate models.

Verified mode compares a local direct/pivot OPUS-MT translation with a free
OpenRouter multilingual translation, then uses a free judge by default.
Fast mode is entirely local and therefore makes no API calls.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
import time
from functools import lru_cache
from pathlib import Path

import requests
from dotenv import load_dotenv

import calculate_cost


PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")

URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEY = os.getenv("OPENROUTER_API_KEY")
FREE_MULTILINGUAL_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
MULTILINGUAL_MODEL = FREE_MULTILINGUAL_MODEL
TRANSLATION_JUDGES = {
    "free": "openrouter/free",
    "lite": "anthropic/claude-haiku-5.5",
    "high": "anthropic/claude-haiku-5.5",
    "max": "openai/gpt-6-astra-pro",
}
ACTIVE_MODEL_PROFILE = "free"
JUDGE_MODEL = TRANSLATION_JUDGES[ACTIVE_MODEL_PROFILE]
USAGE = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
MODEL_USAGE: dict[str, dict] = {}


def configure_model_profile(profile: str) -> None:
    """Select the translation judge independently from the OCR model profile."""
    if profile not in TRANSLATION_JUDGES:
        raise ValueError(f"Unknown model profile {profile!r}")
    global ACTIVE_MODEL_PROFILE, JUDGE_MODEL, MULTILINGUAL_MODEL
    ACTIVE_MODEL_PROFILE = profile
    JUDGE_MODEL = TRANSLATION_JUDGES[profile]
    MULTILINGUAL_MODEL = (
        "openrouter/free" if profile == "free" else FREE_MULTILINGUAL_MODEL
    )

TARGETS = {
    "en": ("English", "en"), "english": ("English", "en"),
    "vi": ("Vietnamese", "vi"), "vietnamese": ("Vietnamese", "vi"),
    "fr": ("French", "fr"), "french": ("French", "fr"),
    "de": ("German", "de"), "german": ("German", "de"),
    "es": ("Spanish", "es"), "spanish": ("Spanish", "es"),
    "it": ("Italian", "it"), "italian": ("Italian", "it"),
    "pt": ("Portuguese", "pt"), "portuguese": ("Portuguese", "pt"),
    "ru": ("Russian", "ru"), "russian": ("Russian", "ru"),
    "zh": ("Chinese", "zh"), "chinese": ("Chinese", "zh"),
    "ja": ("Japanese", "ja"), "japanese": ("Japanese", "ja"),
    "ko": ("Korean", "ko"), "korean": ("Korean", "ko"),
    "tr": ("Turkish", "tr"), "turkish": ("Turkish", "tr"),
    "id": ("Indonesian", "id"), "indonesian": ("Indonesian", "id"),
}

# Purpose-built local Arabic translation models.
SPECIALIST_MODELS = {
    "en": "Helsinki-NLP/opus-mt-ar-en",
    "fr": "Helsinki-NLP/opus-mt-ar-fr",
    "de": "Helsinki-NLP/opus-mt-ar-de",
    "es": "Helsinki-NLP/opus-mt-ar-es",
}

# Local English-pivot routes for targets without a reliable direct Arabic model.
# Each tuple is (Arabic -> English, English -> target, optional target prefix).
PIVOT_MODELS = {
    "vi": (
        "Helsinki-NLP/opus-mt-ar-en",
        "Helsinki-NLP/opus-mt-en-vi",
        None,
    ),
    "it": (
        "Helsinki-NLP/opus-mt-ar-en",
        "Helsinki-NLP/opus-mt-en-it",
        None,
    ),
    "pt": (
        "Helsinki-NLP/opus-mt-ar-en",
        "Helsinki-NLP/opus-mt-tc-big-en-pt",
        ">>por<<",
    ),
    "ru": (
        "Helsinki-NLP/opus-mt-ar-en",
        "Helsinki-NLP/opus-mt-en-ru",
        None,
    ),
    "zh": (
        "Helsinki-NLP/opus-mt-ar-en",
        "Helsinki-NLP/opus-mt-en-zh",
        ">>cmn_Hans<<",
    ),
    "ja": (
        "Helsinki-NLP/opus-mt-ar-en",
        "Helsinki-NLP/opus-mt-en-jap",
        None,
    ),
    "ko": (
        "Helsinki-NLP/opus-mt-ar-en",
        "Helsinki-NLP/opus-mt-tc-big-en-ko",
        None,
    ),
    "tr": (
        "Helsinki-NLP/opus-mt-ar-en",
        "Helsinki-NLP/opus-mt-tc-big-en-tr",
        None,
    ),
    "id": (
        "Helsinki-NLP/opus-mt-ar-en",
        "Helsinki-NLP/opus-mt-en-id",
        None,
    ),
}
LOCAL_TARGET_CODES = tuple(sorted(set(SPECIALIST_MODELS) | set(PIVOT_MODELS)))


def normalize_target(target_language: str) -> tuple[str, str]:
    key = target_language.strip().lower().replace("_", "-")
    if key in TARGETS:
        return TARGETS[key]
    if re.fullmatch(r"[a-z]{2,3}(?:-[a-z]{2,4})?", key):
        return key, key
    raise ValueError(
        "Invalid target language. Use a code such as en, vi, fr, de, es, or a name."
    )


def _require_free_model(model: str) -> None:
    if not (model.endswith(":free") or model == "openrouter/free"):
        raise ValueError(
            f"Translation candidate model must be free, but got {model!r}. "
            "Use an OpenRouter :free model."
        )


def _chat(model: str, prompt: str, retries: int = 3) -> str:
    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 8192,
    }
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    last_error = None
    for attempt in range(retries):
        try:
            response = requests.post(URL, headers=headers, json=payload, timeout=180)
            response.raise_for_status()
            data = response.json()
            if "error" in data:
                raise RuntimeError(data["error"])
            usage = data.get("usage") or {}
            USAGE["calls"] += 1
            USAGE["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
            USAGE["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
            calculate_cost.record_usage(MODEL_USAGE, model, usage)
            return data["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"{model} failed after {retries} tries: {last_error}")


def _strip_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json|text)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _parse_json(text: str) -> dict:
    cleaned = _strip_fence(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _split_text(text: str, max_chars: int = 4200) -> list[str]:
    """Split on lines while retaining OCR location annotations."""
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for line in text.splitlines():
        addition = len(line) + 1
        if current and size + addition > max_chars:
            chunks.append("\n".join(current).strip())
            current, size = [], 0
        if len(line) > max_chars:
            if current:
                chunks.append("\n".join(current).strip())
                current, size = [], 0
            for start in range(0, len(line), max_chars):
                chunks.append(line[start:start + max_chars])
            continue
        current.append(line)
        size += addition
    if current:
        chunks.append("\n".join(current).strip())
    return [chunk for chunk in chunks if chunk]


@lru_cache(maxsize=2)
def _load_specialist(model_name: str):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    except OSError as exc:
        raise RuntimeError(
            f"Could not load local translation model {model_name!r}. The first use "
            "needs an internet connection to download it from Hugging Face; later "
            "runs use the local cache and still make zero OpenRouter API calls."
        ) from exc
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device


def _specialist_translate(
    text: str, model_name: str, source_prefix: str | None = None
) -> str:
    import torch

    tokenizer, model, device = _load_specialist(model_name)

    def translate_unit(unit: str) -> str:
        if source_prefix:
            unit = f"{source_prefix} {unit}"
        token_ids = tokenizer(unit, add_special_tokens=False)["input_ids"]
        pieces = []
        for start in range(0, len(token_ids), 440):
            source_piece = tokenizer.decode(
                token_ids[start:start + 440], skip_special_tokens=True
            )
            encoded = tokenizer(
                source_piece, return_tensors="pt", truncation=True, max_length=480
            ).to(device)
            output_limit = min(480, max(64, len(token_ids[start:start + 440]) * 2 + 24))
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    max_length=output_limit,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.15,
                    renormalize_logits=True,
                )
            pieces.append(
                tokenizer.decode(generated[0], skip_special_tokens=True).strip()
            )
        return " ".join(piece for piece in pieces if piece)

    translated_lines = []
    for raw_line in text.splitlines() or [text]:
        line = raw_line.strip()
        if not line:
            translated_lines.append("")
            continue
        if re.fullmatch(r"\[\[.*\]\]", line):
            translated_lines.append(line)
            continue
        # Short sentence units stop Marian from silently dropping an earlier
        # sentence in a line, which is especially visible in Arabic -> English.
        sentences = re.split(r"(?<=[.!?؟؛])\s+", line)
        translated_lines.append(
            " ".join(translate_unit(sentence) for sentence in sentences if sentence)
        )
    return "\n".join(translated_lines).strip()


def _pivot_translate(
    text: str, model_route: tuple[str, str, str | None]
) -> tuple[str, str]:
    """Translate Arabic through English and return (target, English intermediate)."""
    ar_to_en, en_to_target, target_prefix = model_route
    english = _specialist_translate(text, ar_to_en)
    target = _specialist_translate(english, en_to_target, target_prefix)
    return target, english


def _multilingual_translate(text: str, target_name: str) -> str:
    _require_free_model(MULTILINGUAL_MODEL)
    prompt = f"""Translate the following Arabic document into {target_name}.

Requirements:
- Preserve meaning, names, dates, numbers, headings, and paragraph order.
- Preserve markers such as [[OCR_IMAGE | ...]] exactly; do not translate their contents.
- Do not summarize, explain, censor, or add commentary.
- Return only the translation.

ARABIC SOURCE:
{text}"""
    return _strip_fence(_chat(MULTILINGUAL_MODEL, prompt))


def _judge_translation(
    source: str,
    local_candidate: str,
    multilingual: str,
    target_name: str,
    local_route: str,
) -> dict:
    prompt = f"""You are verifying two translations from Arabic into {target_name}.
Compare both candidates against the Arabic source. Produce the most faithful final
translation. Preserve names, dates, numbers, document structure, and all [[...]]
location markers. Do not summarize or invent missing content.

Return ONLY JSON:
{{
  "translation": "complete final translation",
  "preferred": "local|multilingual|merged",
  "differences": ["brief material difference"],
  "notes": "brief verification note"
}}

ARABIC SOURCE:
{source}

LOCAL {local_route.upper()} CANDIDATE:
{local_candidate}

MULTILINGUAL CANDIDATE:
{multilingual}"""
    result = _parse_json(_chat(JUDGE_MODEL, prompt))
    translation = str(result.get("translation", "")).strip()
    if not translation:
        raise ValueError("Translation judge returned no final translation")
    return {
        "translation": translation,
        "preferred": str(result.get("preferred", "merged")),
        "differences": result.get("differences", []),
        "notes": str(result.get("notes", "")),
    }


def translate_with_report(
    text: str, target_language: str, mode: str = "verified"
) -> tuple[str, dict]:
    """Translate Arabic text and return both final text and an audit report."""
    if mode not in {"verified", "fast"}:
        raise ValueError("mode must be 'verified' or 'fast'")
    if not text.strip():
        raise ValueError("Cannot translate empty text")

    usage_before = dict(USAGE)
    target_name, target_code = normalize_target(target_language)
    specialist_model = SPECIALIST_MODELS.get(target_code)
    pivot_models = PIVOT_MODELS.get(target_code)
    if specialist_model:
        local_route = "direct_local"
    elif pivot_models:
        local_route = "pivot_local"
    else:
        local_route = None

    if mode == "fast" and local_route is None:
        raise ValueError(
            f"Fast translation to {target_code!r} is unavailable: no local direct "
            "or English-pivot route is configured. Fast mode never calls an API. "
            f"Configured local targets: {', '.join(LOCAL_TARGET_CODES)}."
        )

    effective_mode = "verified" if mode == "verified" and local_route else mode
    chunks = _split_text(text)
    final_chunks: list[str] = []
    local_chunks: list[str] = []
    pivot_english_chunks: list[str] = []
    multilingual_chunks: list[str] = []
    chunk_reports: list[dict] = []

    for index, source_chunk in enumerate(chunks, start=1):
        local_candidate = None
        if local_route == "direct_local":
            local_candidate = _specialist_translate(source_chunk, specialist_model)
        elif local_route == "pivot_local":
            local_candidate, pivot_english = _pivot_translate(source_chunk, pivot_models)
            pivot_english_chunks.append(pivot_english)
        if local_candidate is not None:
            local_chunks.append(local_candidate)

        if mode == "fast":
            final = local_candidate
            chunk_reports.append({
                "chunk": index,
                "preferred": "local",
                "differences": [],
                "notes": "Fast mode used the local route; no API calls were made.",
            })
        elif local_candidate is not None:
            multilingual = _multilingual_translate(source_chunk, target_name)
            multilingual_chunks.append(multilingual)
            judged = _judge_translation(
                source_chunk,
                local_candidate,
                multilingual,
                target_name,
                local_route,
            )
            final = judged.pop("translation")
            similarity = difflib.SequenceMatcher(
                None, local_candidate, multilingual
            ).ratio()
            chunk_reports.append({
                "chunk": index,
                "candidate_similarity": round(similarity, 4),
                **judged,
            })
        else:
            multilingual = _multilingual_translate(source_chunk, target_name)
            multilingual_chunks.append(multilingual)
            final = multilingual
            chunk_reports.append({
                "chunk": index,
                "preferred": "multilingual",
                "differences": [],
                "notes": "No configured local route for this target; used multilingual only.",
            })
        final_chunks.append(final)

    final_text = "\n\n".join(final_chunks).strip()
    report = {
        "model_profile": ACTIVE_MODEL_PROFILE,
        "source_language": "ar",
        "target_language": target_code,
        "target_name": target_name,
        "requested_mode": mode,
        "effective_mode": effective_mode,
        "translation_route": local_route or "multilingual_only",
        "models": {
            "local_direct": specialist_model,
            "pivot_ar_to_en": pivot_models[0] if pivot_models else None,
            "pivot_en_to_target": pivot_models[1] if pivot_models else None,
            "pivot_target_prefix": pivot_models[2] if pivot_models else None,
            "multilingual": MULTILINGUAL_MODEL if mode == "verified" else None,
            "judge": JUDGE_MODEL if effective_mode == "verified" else None,
        },
        "candidate_models_are_free": True,
        "chunks": chunk_reports,
        "usage": {
            key: USAGE[key] - usage_before[key]
            for key in USAGE
        },
        "local_candidate_text": "\n\n".join(local_chunks) if local_chunks else None,
        "pivot_english_text": (
            "\n\n".join(pivot_english_chunks) if pivot_english_chunks else None
        ),
        "multilingual_text": (
            "\n\n".join(multilingual_chunks) if multilingual_chunks else None
        ),
    }
    return final_text, report


def translate_text(text: str, target_language: str, mode: str = "verified") -> str:
    """Translate Arabic text into the requested language."""
    translated, _ = translate_with_report(text, target_language, mode=mode)
    return translated


def translate_file(
    input_path: str | Path,
    target_language: str,
    *,
    mode: str = "verified",
    output_path: str | Path,
    log_dir: str | Path,
) -> Path:
    input_path, output_path, log_dir = Path(input_path), Path(output_path), Path(log_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    translated, report = translate_with_report(
        input_path.read_text(encoding="utf-8"), target_language, mode=mode
    )
    output_path.write_text(translated, encoding="utf-8")
    multilingual = report.pop("multilingual_text")
    local_candidate = report.pop("local_candidate_text")
    pivot_english = report.pop("pivot_english_text")
    if multilingual is not None:
        (log_dir / "multilingual.txt").write_text(multilingual, encoding="utf-8")
    if local_candidate is not None:
        (log_dir / "local_candidate.txt").write_text(
            local_candidate, encoding="utf-8"
        )
    if pivot_english is not None:
        (log_dir / "pivot_english.txt").write_text(
            pivot_english, encoding="utf-8"
        )
    report.update({"input": str(input_path), "output": str(output_path)})
    (log_dir / "translation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output_path


def main_cli() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Translate Arabic text or a UTF-8 text file.")
    parser.add_argument("source", help="literal Arabic text or path to a UTF-8 text file")
    parser.add_argument("target_language", help="target code/name, for example en or vi")
    parser.add_argument(
        "--mode",
        choices=["verified", "fast"],
        default="verified",
        help="verified: compare local and multilingual candidates; fast: local only, zero API calls",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--log-dir", type=Path)
    profile_group = parser.add_mutually_exclusive_group()
    for profile in ("free", "lite", "high", "max"):
        profile_group.add_argument(
            f"--{profile}",
            dest="model_profile",
            action="store_const",
            const=profile,
            help=f"use the {profile} translation profile (default: free)",
        )
    parser.set_defaults(model_profile="free")
    args = parser.parse_args()
    configure_model_profile(args.model_profile)

    source_path = Path(args.source)
    if source_path.is_file():
        _, target_code = normalize_target(args.target_language)
        output = args.output or source_path.with_name(f"{source_path.stem}_{target_code}.txt")
        log_dir = args.log_dir or source_path.with_name(f"{source_path.stem}_{target_code}_log")
        result = translate_file(
            source_path,
            args.target_language,
            mode=args.mode,
            output_path=output,
            log_dir=log_dir,
        )
        print(f"Translation: {result}")
        print(f"Translation log: {log_dir}")
    else:
        print(translate_text(args.source, args.target_language, mode=args.mode))
    print(
        f"API calls: {USAGE['calls']}, prompt tokens: {USAGE['prompt_tokens']}, "
        f"completion tokens: {USAGE['completion_tokens']}"
    )
    calculate_cost.print_cost_report(
        calculate_cost.build_cost_report(
            {"translation": MODEL_USAGE},
            command=sys.argv,
        )
    )


if __name__ == "__main__":
    main_cli()
