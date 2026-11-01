"""Complete local-first OCR implementation for the ``--free`` profile.

This module owns both local candidates (fine-tuned KHATT CRNN-CTC and Nuqta),
the constrained OpenRouter diacritic pass, and deterministic A/B fallback.
Keeping the free profile together makes its network boundary explicit: images
remain local; only OCR text may be sent to the configured free resolver.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image
from dotenv import load_dotenv

import calculate_cost


PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")

CRNN_MODEL_NAME = "FixFips/arabicocr-khatt (CRNN-CTC)"
NUQTA_MODEL_NAME = "Nuqta Arabic OCR 0.8B Q4"
DEFAULT_FINETUNED_DIR = PROJECT_DIR / "models" / "arabicocr-khatt-finetuned"
_OCR = None
_LOAD_ERROR: Exception | None = None


@dataclass
class LocalOCRResult:
    engine: str
    lines: list[str]
    elapsed_seconds: float
    error: str | None = None
    segmentation: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def _clean_lines(lines: list[str]) -> list[str]:
    return [
        re.sub(r"[ \t]+", " ", line).strip()
        for line in lines
        if isinstance(line, str) and line.strip()
    ]


def _load_model():
    global CRNN_MODEL_NAME, _OCR, _LOAD_ERROR
    if _OCR is not None:
        return _OCR
    if _LOAD_ERROR is not None:
        raise RuntimeError(str(_LOAD_ERROR)) from _LOAD_ERROR
    try:
        from arabicocr_khatt import ArabicOCR

        device = os.getenv("ARABICOCR_KHATT_DEVICE") or None
        configured_checkpoint = os.getenv("ARABICOCR_KHATT_CHECKPOINT")
        checkpoint = (
            Path(configured_checkpoint).expanduser().resolve()
            if configured_checkpoint
            else DEFAULT_FINETUNED_DIR / "crnn_best.pt"
        )
        if configured_checkpoint and not checkpoint.is_file():
            raise FileNotFoundError(
                f"ARABICOCR_KHATT_CHECKPOINT does not exist: {checkpoint}"
            )
        if checkpoint.is_file():
            bigram_path = checkpoint.with_name("bigram_lm.json")
            bigram_lm = None
            if bigram_path.is_file():
                from arabicocr_khatt.pipeline import load_bigram_lm_json

                bigram_lm = load_bigram_lm_json(bigram_path)
            _OCR = ArabicOCR(str(checkpoint), device=device, bigram_lm=bigram_lm)
            CRNN_MODEL_NAME = f"Fine-tuned ArabicOCR_KHATT ({checkpoint.name})"
        else:
            _OCR = ArabicOCR.from_pretrained(device=device)
        return _OCR
    except Exception as exc:
        _LOAD_ERROR = exc
        raise RuntimeError(
            "ArabicOCR_KHATT is unavailable. Install it with "
            "`python -m pip install arabicocr-khatt`. The first run also needs "
            "network access to download its checkpoint. A fine-tuned checkpoint "
            "can be selected with ARABICOCR_KHATT_CHECKPOINT. Original error: "
            f"{exc}"
        ) from exc


def _segment_tall_page(img: Image.Image) -> list[Image.Image]:
    """Avoid the upstream page segmenter's height-scaled over-merging."""
    if img.height <= img.width * 1.2:
        return [img]

    from arabicocr_khatt.pipeline import _robust_binarize, segment_into_lines

    binary = _robust_binarize(img)
    ink = 255 - binary
    height, width = ink.shape
    rule_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(width // 8, 80), 1)
    )
    rules = cv2.morphologyEx(ink, cv2.MORPH_OPEN, rule_kernel, iterations=1)
    projection = (cv2.subtract(ink, rules) > 0).sum(axis=1)

    target_height = max(140, min(220, int(width * 0.18)))
    cuts = [0]
    cursor = 0
    while height - cursor > target_height + 60:
        ideal = cursor + target_height
        low = max(cursor + 110, ideal - 40)
        high = min(height - 80, ideal + 40)
        if high <= low:
            break
        cut = low + int(np.argmin(projection[low : high + 1]))
        cuts.append(cut)
        cursor = cut
    cuts.append(height)

    lines: list[Image.Image] = []
    for top, bottom in zip(cuts, cuts[1:]):
        tile = img.crop((0, top, width, bottom))
        detected = segment_into_lines(tile, min_h=10, min_width_ratio=0.15)
        # segment_into_lines returns its full input when no candidate survives.
        if len(detected) == 1 and detected[0].size == tile.size:
            continue
        lines.extend(detected)
    return lines or [img]


def recognize_crnn(img: Image.Image) -> LocalOCRResult:
    """Recognize a raw page with the single KHATT CRNN-CTC model."""
    started = time.perf_counter()
    try:
        model = _load_model()
        configured_upscale = os.getenv("ARABICOCR_KHATT_UPSCALE")
        upscale = (
            max(1.0, float(configured_upscale))
            if configured_upscale
            else (2.0 if img.width < 1500 else 1.0)
        )
        rgb = img.convert("RGB")
        line_images = _segment_tall_page(rgb)
        tiled = not (len(line_images) == 1 and line_images[0] is rgb)
        if tiled:
            recognized = [
                model.recognize(line, segment=False, upscale=upscale)
                for line in line_images
            ]
        else:
            recognized = model.recognize_lines(rgb, segment=True, upscale=upscale)
        lines = _clean_lines(recognized)
        if not lines:
            raise RuntimeError("ArabicOCR_KHATT returned no text")
        return LocalOCRResult(
            CRNN_MODEL_NAME,
            lines,
            time.perf_counter() - started,
            segmentation="tiled_line_segmentation" if tiled else "upstream_page_segmentation",
        )
    except Exception as exc:
        return LocalOCRResult(
            CRNN_MODEL_NAME, [], time.perf_counter() - started, str(exc)
        )


# ---------------------------------------------------------------------------
# Nuqta local candidate

NUQTA_MODEL_REPO = "SalahALHaismawi/Nuqta-Arabic-OCR-0.8B-GGUF:Q4_K_M"
NUQTA_PROMPT = "Transcribe the Arabic text in this image without any diacritics."


@dataclass
class NuqtaResult:
    engine: str
    lines: list[str]
    elapsed_seconds: float
    error: str | None = None
    executable: str | None = None
    loop_detected: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


def _find_nuqta_executable() -> Path:
    configured = os.getenv("NUQTA_LLAMA_CLI")
    if configured:
        path = Path(configured).expanduser()
        if path.is_file():
            return path.resolve()
        raise FileNotFoundError(f"NUQTA_LLAMA_CLI does not exist: {path}")

    discovered = shutil.which("llama-mtmd-cli")
    if discovered:
        return Path(discovered).resolve()

    winget_roots = [
        Path(os.getenv("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages",
        Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages",
    ]
    for winget_root in winget_roots:
        if winget_root.is_dir():
            matches = sorted(winget_root.glob("ggml.llamacpp_*/llama-mtmd-cli.exe"))
            if matches:
                return matches[-1].resolve()
    raise FileNotFoundError(
        "llama-mtmd-cli was not found; install it with "
        "`winget install --id ggml.llamacpp --exact` or set NUQTA_LLAMA_CLI"
    )


def _has_repetition_loop(text: str) -> bool:
    compact = re.sub(r"\s+", " ", text)
    if re.search(r"(.)\1{12,}", compact):
        return True
    return bool(re.search(r"(.{2,20})(?:\1){7,}", compact))


def _clean_nuqta_output(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = text.replace("�", "").strip()
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def recognize_nuqta(img: Image.Image) -> NuqtaResult:
    """Read one page with Nuqta, rejecting empty or repeating output."""
    started = time.perf_counter()
    executable = None
    temp_path = None
    try:
        executable = _find_nuqta_executable()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            temp_path = Path(handle.name)
        img.convert("RGB").save(temp_path, format="PNG")

        command = [
            str(executable),
            "-hf",
            os.getenv("NUQTA_MODEL_REPO", NUQTA_MODEL_REPO),
            "--image",
            str(temp_path),
            "--temp",
            "0",
            "--top-k",
            "1",
            "-c",
            "8192",
            "--image-max-tokens",
            os.getenv("NUQTA_IMAGE_MAX_TOKENS", "4096"),
            "-n",
            os.getenv("NUQTA_MAX_TOKENS", "1024"),
            "--no-warmup",
            "-p",
            NUQTA_PROMPT,
        ]
        if os.getenv("NUQTA_OFFLINE", "").lower() in {"1", "true", "yes"}:
            command.insert(3, "--offline")
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=float(os.getenv("NUQTA_TIMEOUT", "180")),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip().splitlines()
            raise RuntimeError(detail[-1] if detail else f"exit code {completed.returncode}")
        text = _clean_nuqta_output(completed.stdout)
        if not text:
            raise RuntimeError("Nuqta returned no transcription")
        if _has_repetition_loop(text):
            return NuqtaResult(
                NUQTA_MODEL_NAME,
                [],
                time.perf_counter() - started,
                error="Nuqta repetition loop detected; candidate rejected",
                executable=executable.name,
                loop_detected=True,
            )
        return NuqtaResult(
            NUQTA_MODEL_NAME,
            text.splitlines(),
            time.perf_counter() - started,
            executable=executable.name,
        )
    except subprocess.TimeoutExpired:
        return NuqtaResult(
            NUQTA_MODEL_NAME,
            [],
            time.perf_counter() - started,
            error="Nuqta exceeded its local inference timeout",
            executable=executable.name if executable else None,
        )
    except Exception as exc:
        return NuqtaResult(
            NUQTA_MODEL_NAME,
            [],
            time.perf_counter() - started,
            error=str(exc),
            executable=executable.name if executable else None,
        )
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Constrained free text post-processing

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
ROUTER_MODEL = "openrouter/free"
RESOLVER_MODEL = "nvidia/nemotron-3.5-super-vl:free"
DEFAULT_RESOLVER_MODELS = (
    RESOLVER_MODEL,
    "thinkingmachines/inkling:free",
    "google/gemma-4-26b-a4b-it:free",
    ROUTER_MODEL,
)
RESOLVER_MODELS = tuple(
    model.strip()
    for model in os.getenv(
        "OPENROUTER_FREE_MODELS", ",".join(DEFAULT_RESOLVER_MODELS)
    ).split(",")
    if model.strip()
) or DEFAULT_RESOLVER_MODELS
MODEL_USAGE: dict[str, dict] = {}
REQUEST_TIMEOUT_SECONDS = float(os.getenv("OPENROUTER_FREE_TIMEOUT", "45"))

DIACRITIC_PROMPT = """You are a strictly constrained Arabic diacritic editor.
Review the OCR transcript below in context and correct Arabic vowel marks
(tashkeel/harakat) only.

Hard constraints:
- You may add, remove, or replace Unicode combining marks only.
- Never change, insert, delete, or reorder a base letter, digit, punctuation
  mark, space, newline, name, or word.
- Preserve the exact number and order of lines.
- Do not rewrite spelling, normalize alef/hamza, explain, or translate.
- Add marks only where context makes them sufficiently certain; leaving text
  unvocalized is better than guessing.

Return ONLY JSON in this shape:
{"text":"the full transcript with literal newlines escaped by JSON"}

OCR engine: {source}
Transcript:
"""

RESOLVE_PROMPT = """You are resolving two local Arabic OCR candidates.
For every numbered block, choose A or B using Arabic context and consistency.

Hard constraints:
- Return choices only. Never rewrite, correct, merge, or invent text.
- A choice means the program will copy that candidate verbatim.
- Prefer the candidate with plausible Arabic words, names, numbers, and syntax.
- Preserve uncertain source spelling rather than guessing a third alternative.

Return ONLY JSON: {"choices":["A","B",...]}
The choices array must contain exactly one choice for every block.

Blocks:
"""


@dataclass
class DiacritizationResult:
    text: str
    accepted: bool
    status: str
    model: str = RESOLVER_MODEL
    error: str | None = None
    elapsed_seconds: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class ResolutionResult:
    lines: list[str]
    choices: list[str]
    accepted: bool
    status: str
    model: str = RESOLVER_MODEL
    error: str | None = None
    elapsed_seconds: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


def strip_combining_marks(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    return unicodedata.normalize(
        "NFC", "".join(char for char in decomposed if not unicodedata.combining(char))
    )


def _parse_object(text: str) -> dict:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("provider returned empty content")
    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    start = cleaned.find("{")
    if start < 0:
        raise ValueError("provider response contains no JSON object")
    value, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    if not isinstance(value, dict):
        raise ValueError("provider JSON is not an object")
    return value


def _message_text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(block.get("text", ""))
            for block in content
            if isinstance(block, dict)
            and block.get("type") in {"text", "output_text"}
        )
    return ""


def _request_json(
    prompt: str, retries: int, operation: str, *, max_tokens: int
) -> dict:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    last_error: Exception | None = None
    for attempt in range(retries):
        requested_model = RESOLVER_MODELS[attempt % len(RESOLVER_MODELS)]
        payload = {
            "model": requested_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "provider": {"allow_fallbacks": True},
            "reasoning": (
                {"effort": "minimal"}
                if requested_model == ROUTER_MODEL
                else {"enabled": False}
            ),
        }
        try:
            response = requests.post(
                OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            if not response.ok:
                detail = response.text.replace("\n", " ").strip()[:500]
                raise RuntimeError(
                    f"HTTP {response.status_code} from {requested_model}: {detail}"
                )
            data = response.json()
            if data.get("error"):
                raise RuntimeError(str(data["error"]))
            choices = data.get("choices")
            if not isinstance(choices, list) or not choices:
                raise ValueError("provider response contains no choices")
            choice = choices[0]
            actual_model = str(data.get("model") or requested_model)
            calculate_cost.record_usage(
                MODEL_USAGE, actual_model, data.get("usage") or {}
            )
            content = _message_text(choice.get("message") or {})
            if not content.strip():
                usage = data.get("usage") or {}
                raise ValueError(
                    "empty content"
                    f" (requested={requested_model}, actual={actual_model},"
                    f" finish_reason={choice.get('finish_reason')!r},"
                    f" completion_tokens={usage.get('completion_tokens')!r})"
                )
            return _parse_object(content)
        except Exception as exc:
            last_error = exc
            if attempt + 1 < retries:
                next_model = RESOLVER_MODELS[(attempt + 1) % len(RESOLVER_MODELS)]
                print(
                    f"{requested_model} {operation} failed ({exc}); "
                    f"retrying with {next_model}",
                    file=sys.stderr,
                )
                time.sleep(2**attempt)
    raise RuntimeError(str(last_error))


def _candidate_quality(text: str) -> float:
    compact = re.sub(r"\s+", " ", strip_combining_marks(text)).strip()
    if not compact:
        return -math.inf
    visible = [char for char in compact if not char.isspace()]
    arabic = re.findall(r"[\u0600-\u06ff]", compact)
    words = re.findall(r"[\u0621-\u064a]+", compact)
    unique_ratio = len(set(words)) / max(1, len(words))
    arabic_ratio = len(arabic) / max(1, len(visible))
    coverage = min(1.0, math.log1p(len(arabic)) / math.log(401))
    bad = compact.count("�") + len(re.findall(r"(.)\1{7,}", compact)) * 8
    latin_noise = len(re.findall(r"[A-Za-z]", compact)) / max(1, len(visible))
    return (
        arabic_ratio * 3.0
        + unique_ratio
        + coverage * 0.6
        - bad / max(1, len(visible)) * 4.0
        - latin_noise * 2.0
    )


def _local_fallback_choice(block: dict) -> str:
    left = str(block.get("a") or "")
    right = str(block.get("b") or "")
    if not left.strip():
        return "B" if right.strip() else "A"
    if not right.strip():
        return "A"
    margin = 0.05 if block.get("status") == "whole_document_choice" else 0.15
    return "B" if _candidate_quality(right) > _candidate_quality(left) + margin else "A"


def correct_diacritics(
    text: str, *, source: str = "local OCR", retries: int = 2
) -> DiacritizationResult:
    started = time.perf_counter()
    if not text.strip():
        return DiacritizationResult(text, False, "skipped_empty")
    if not OPENROUTER_API_KEY:
        return DiacritizationResult(
            text,
            False,
            "fallback_no_api_key",
            error="OPENROUTER_API_KEY is not set",
            elapsed_seconds=time.perf_counter() - started,
        )
    try:
        value = _request_json(
            DIACRITIC_PROMPT.replace("{source}", source) + text,
            retries,
            "diacritic pass",
            max_tokens=max(256, min(4096, len(text) * 2)),
        )
        candidate = value.get("text")
        if not isinstance(candidate, str) or not candidate:
            raise ValueError("provider JSON does not contain a non-empty text field")
        if strip_combining_marks(candidate) != strip_combining_marks(text):
            return DiacritizationResult(
                text,
                False,
                "rejected_base_text_change",
                error="response changed non-diacritic content",
                elapsed_seconds=time.perf_counter() - started,
            )
        return DiacritizationResult(
            candidate,
            True,
            "accepted" if candidate != text else "accepted_unchanged",
            elapsed_seconds=time.perf_counter() - started,
        )
    except Exception as exc:
        return DiacritizationResult(
            text,
            False,
            "fallback_api_error",
            error=str(exc),
            elapsed_seconds=time.perf_counter() - started,
        )


def resolve_candidates(blocks: list[dict], retries: int = 2) -> ResolutionResult:
    started = time.perf_counter()
    if not blocks:
        return ResolutionResult([], [], False, "skipped_empty")

    def materialize(choices: list[str]) -> list[str]:
        lines = []
        for block, choice in zip(blocks, choices):
            selected = block.get(choice.lower(), "")
            if not selected:
                selected = block.get("a") or block.get("b") or ""
            if selected:
                lines.append(selected)
        return lines

    fallback_choices = [_local_fallback_choice(block) for block in blocks]
    fallback_lines = materialize(fallback_choices)
    if not OPENROUTER_API_KEY:
        return ResolutionResult(
            fallback_lines,
            fallback_choices,
            False,
            "fallback_no_api_key",
            error="OPENROUTER_API_KEY is not set",
            elapsed_seconds=time.perf_counter() - started,
        )

    prompt_blocks = [
        {"index": index, "A": block.get("a", ""), "B": block.get("b", "")}
        for index, block in enumerate(blocks, start=1)
    ]
    try:
        value = _request_json(
            RESOLVE_PROMPT + json.dumps(prompt_blocks, ensure_ascii=False, indent=2),
            retries,
            "candidate resolution",
            max_tokens=max(128, min(1024, len(blocks) * 8 + 64)),
        )
        choices = value.get("choices")
        if (
            not isinstance(choices, list)
            or len(choices) != len(blocks)
            or any(str(choice).upper() not in {"A", "B"} for choice in choices)
        ):
            raise ValueError("resolver returned an invalid choices array")
        choices = [str(choice).upper() for choice in choices]
        return ResolutionResult(
            materialize(choices),
            choices,
            True,
            "accepted",
            elapsed_seconds=time.perf_counter() - started,
        )
    except Exception as exc:
        return ResolutionResult(
            fallback_lines,
            fallback_choices,
            False,
            "fallback_api_error",
            error=str(exc),
            elapsed_seconds=time.perf_counter() - started,
        )
