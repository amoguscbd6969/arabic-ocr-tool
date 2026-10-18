"""Paid OpenRouter OCR services.

This module owns model selection, prompts, remote calls, classification,
transcription, crop adjudication, and visual rechecks.  ``main.py`` only
orchestrates input, local preprocessing, pipeline flow, logs, and output.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from PIL import Image

import calculate_cost
import preprocess


load_dotenv(Path(__file__).parent / ".env")

API_KEY = os.getenv("OPENROUTER_API_KEY")
URL = "https://openrouter.ai/api/v1/chat/completions"
PAID_MODEL_PROFILES = {
    "lite": {
        "classifier": "anthropic/claude-haiku-5.5",
        "printed": "anthropic/claude-haiku-5.5",
        "handwritten": "anthropic/claude-haiku-5.5",
        "secondary": "openai/gpt-6-luna",
        "recheck": "anthropic/claude-haiku-5.5",
        "crop": "anthropic/claude-haiku-5.5",
        "crop_judge": "anthropic/claude-haiku-5.5",
        "mark_blind": "anthropic/claude-haiku-5.5",
        "mark_forensic": "openai/gpt-6-luna",
        "mark_judge": "anthropic/claude-haiku-5.5",
    },
    "high": {
        "classifier": "anthropic/claude-haiku-5.5",
        "printed": "anthropic/claude-haiku-5.5",
        "handwritten": "anthropic/claude-sonnet-5.5",
        "secondary": "openai/gpt-6-luna",
        "recheck": "anthropic/claude-sonnet-5.5",
        "crop": "anthropic/claude-haiku-5.5",
        "crop_judge": "anthropic/claude-haiku-5.5",
        "mark_blind": "anthropic/claude-sonnet-5.5",
        "mark_forensic": "anthropic/claude-sonnet-5.5",
        "mark_judge": "anthropic/claude-opus-5.5",
    },
    "max": {
        "classifier": "anthropic/claude-opus-5.5",
        "printed": "anthropic/claude-opus-5.5",
        "handwritten": "anthropic/claude-opus-5.5",
        "secondary": "openai/gpt-6-astra-pro",
        "recheck": "anthropic/claude-opus-5.5",
        "crop": "anthropic/claude-opus-5.5",
        "crop_judge": "openai/gpt-6-astra-pro",
        "mark_blind": "anthropic/claude-opus-5.5",
        "mark_forensic": "openai/gpt-6-astra-pro",
        "mark_judge": "openai/gpt-6-astra-pro",
    },
}

ACTIVE_MODEL_PROFILE = "lite"
USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
MODEL_USAGE: dict[str, dict] = {}
UNCLEAR_RE = re.compile(r"[\[\(\{]\s*غير\s+واضح\s*[\]\)\}]")


class OpenRouterResponseError(RuntimeError):
    """Represent an error returned inside a successful HTTP response."""

    def __init__(self, details) -> None:
        self.details = details
        self.status_code = details.get("code") if isinstance(details, dict) else None
        message = details.get("message", str(details)) if isinstance(details, dict) else str(details)
        super().__init__(message)


def configure_model_profile(profile: str) -> None:
    """Select one paid tier, or disable remote OCR for the free tier."""
    global ACTIVE_MODEL_PROFILE
    global MODEL_CLASSIFIER, MODEL_PRINTED, MODEL_HANDWRITTEN, MODEL_SECONDARY
    global MODEL_RECHECK, MODEL_CROP, MODEL_CROP_JUDGE
    global MODEL_MARK_BLIND, MODEL_MARK_FORENSIC, MODEL_MARK_JUDGE

    ACTIVE_MODEL_PROFILE = profile
    if profile == "free":
        models = {key: None for key in next(iter(PAID_MODEL_PROFILES.values()))}
    else:
        if profile not in PAID_MODEL_PROFILES:
            raise ValueError(f"Unknown model profile {profile!r}")
        models = PAID_MODEL_PROFILES[profile]
    MODEL_CLASSIFIER = models["classifier"]
    MODEL_PRINTED = models["printed"]
    MODEL_HANDWRITTEN = models["handwritten"]
    MODEL_SECONDARY = models["secondary"]
    MODEL_RECHECK = models["recheck"]
    MODEL_CROP = models["crop"]
    MODEL_CROP_JUDGE = models["crop_judge"]
    MODEL_MARK_BLIND = models["mark_blind"]
    MODEL_MARK_FORENSIC = models["mark_forensic"]
    MODEL_MARK_JUDGE = models["mark_judge"]


configure_model_profile(ACTIVE_MODEL_PROFILE)


def to_data_url(img: Image.Image, fmt: str = "PNG") -> str:
    """Convert a PIL image to an RGB data URL for OpenRouter."""
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[-1])
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")
    buffer = io.BytesIO()
    img.save(buffer, format=fmt)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{encoded}"


def parse_json(text: str):
    """Extract the first JSON array or object from a possibly fenced reply."""
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    starts = [position for position in (text.find("["), text.find("{")) if position >= 0]
    if not starts:
        raise ValueError("no JSON found in reply")
    value, _ = json.JSONDecoder().raw_decode(text[min(starts):])
    return value


def _message_text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        result = content.strip()
    elif isinstance(content, list):
        result = "\n".join(
            str(block.get("text", "")).strip()
            for block in content
            if isinstance(block, dict) and block.get("type") in {"text", "output_text"}
        ).strip()
    else:
        result = ""
    if not result:
        raise ValueError("provider returned an empty/null message.content")
    return result


def chat(model: str, text: str, images: list[Image.Image], retries: int = 3) -> str:
    """Call OpenRouter with retry and cost accounting."""
    if ACTIVE_MODEL_PROFILE == "free" or not model:
        raise RuntimeError("The --free OCR profile does not permit OpenRouter calls")
    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    content = [{"type": "text", "text": text}]
    content.extend(
        {"type": "image_url", "image_url": {"url": to_data_url(image)}}
        for image in images
    )
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    last_error = None
    for attempt in range(retries):
        payload = {"model": model, "messages": [{"role": "user", "content": content}]}
        try:
            response = requests.post(URL, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            if "error" in data:
                raise OpenRouterResponseError(data["error"])
            usage = data.get("usage") or {}
            USAGE["prompt_tokens"] += usage.get("prompt_tokens", 0)
            USAGE["completion_tokens"] += usage.get("completion_tokens", 0)
            USAGE["calls"] += 1
            calculate_cost.record_usage(MODEL_USAGE, model, usage)
            answer = _message_text(data["choices"][0]["message"])
            parse_json(answer)
            return answer
        except Exception as exc:
            last_error = RuntimeError(f"{model}: {exc}")
            if attempt + 1 < retries:
                print(f"Model {model} failed ({exc}); retrying", file=sys.stderr)
                time.sleep(2 ** attempt)
    raise RuntimeError(f"{model} failed across {retries} attempt(s): {last_error}")


CLASSIFY_PROMPT = """Classify the Arabic document image by how its text was made.
Return ONLY JSON in this exact shape:
{"type":"printed|handwritten|mixed","confidence":0.0}
Use "printed" only when text to transcribe is predominantly typeset. Use
"handwritten" for predominantly handwriting, and "mixed" when both matter."""

OCR_PROMPT = """Transcribe ALL Arabic text in this image exactly as written.
- Keep diacritics if present. Do not correct spelling, translate, or infer missing words.
- Use one JSON string per visual line, top to bottom, in natural reading order.
- If a word truly cannot be read, write [غير واضح] only for that word.
- Return ONLY a JSON array of strings, e.g. ["السطر الأول", "السطر الثاني"]"""

CROP_CHOICE_PROMPT = """Compare two OCR readings of the same printed Arabic page.
Candidate FULL was read from the entire enhanced page. Candidate CROPS was read
from detected paragraph crops. Inspect the supplied page image as ground truth.

Prefer the candidate that preserves every header, paragraph, footnote, number,
and line in correct reading order with fewer invented or unreadable words. Do
not prefer CROPS merely because it is longer or cleaner. Return ONLY JSON:
{{"choice":"full|crops","confidence":0.0,"reason":"short reason"}}

FULL:
{full}

CROPS:
{crops}"""

RECHECK_PROMPT = """Verify exactly ONE visual line of an Arabic document.
Candidate A: {a}
Candidate B: {b}

Read only the supplied line crop. Return ONLY JSON:
{{"supported":true,"text":"exact transcription"}}
If the crop is not the claimed line or is too unclear, return:
{{"supported":false,"text":""}}"""


def classify_document(img: Image.Image) -> dict:
    result = parse_json(chat(MODEL_CLASSIFIER, CLASSIFY_PROMPT, [img]))
    page_type = str(result.get("type", "mixed")).strip().lower()
    if page_type not in {"printed", "handwritten", "mixed"}:
        page_type = "mixed"
    try:
        confidence = max(0.0, min(1.0, float(result.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    if page_type == "printed" and confidence < 0.70:
        page_type = "mixed"
    return {"type": page_type, "confidence": confidence}


def select_primary_model(page_type: str) -> str:
    return MODEL_PRINTED if page_type == "printed" else MODEL_HANDWRITTEN


def ocr(model: str, img: Image.Image) -> list[str]:
    reply = parse_json(chat(model, OCR_PROMPT, [img]))
    if not isinstance(reply, list):
        raise ValueError(f"{model} returned JSON, but it was not an array")
    lines = []
    for item in reply:
        text = str(item.get("text", "") if isinstance(item, dict) else item).strip()
        if text:
            lines.append(text)
    return lines


def _fallback_crop_choice(full: list[str], crops: list[str]) -> tuple[str, str]:
    full_text, crop_text = "\n".join(full), "\n".join(crops)
    full_unclear = len(UNCLEAR_RE.findall(full_text))
    crop_unclear = len(UNCLEAR_RE.findall(crop_text))
    full_arabic = len(re.findall(r"[\u0600-\u06FF]", full_text))
    crop_arabic = len(re.findall(r"[\u0600-\u06FF]", crop_text))
    if crop_arabic >= full_arabic * 0.90 and crop_unclear < full_unclear:
        return "crops", "local fallback: comparable coverage with fewer unclear spans"
    return "full", "local fallback: crop improvement was not proven"


def printed_ocr_with_optional_crops(
    img: Image.Image,
    prepared: preprocess.PreprocessResult,
    boxes: list[preprocess.ParagraphBox],
    layout_quality: preprocess.LayoutQuality,
) -> tuple[list[str], dict]:
    full_lines = ocr(MODEL_PRINTED, img)
    metadata = {
        "attempted": False,
        "selected": "full",
        "full_lines": len(full_lines),
        "crop_lines": 0,
        "judge": None,
    }
    if not layout_quality.passed:
        metadata["reason"] = f"crop skipped: {layout_quality.reason}"
        return full_lines, metadata

    crop_lines = []
    try:
        for crop in preprocess.crop_paragraphs(prepared, boxes):
            crop_lines.extend(ocr(MODEL_CROP, crop))
    except Exception as exc:
        metadata.update({"attempted": True, "reason": f"crop OCR failed: {exc}"})
        return full_lines, metadata
    metadata.update({"attempted": True, "crop_lines": len(crop_lines)})
    if not crop_lines:
        metadata["reason"] = "crop OCR returned no lines"
        return full_lines, metadata

    prompt = CROP_CHOICE_PROMPT.format(
        full=json.dumps(full_lines, ensure_ascii=False),
        crops=json.dumps(crop_lines, ensure_ascii=False),
    )
    try:
        decision = parse_json(chat(MODEL_CROP_JUDGE, prompt, [img]))
        choice = str(decision.get("choice", "full")).lower()
        confidence = max(0.0, min(1.0, float(decision.get("confidence", 0.0))))
        if choice not in {"full", "crops"} or confidence < 0.60:
            choice, reason = _fallback_crop_choice(full_lines, crop_lines)
        else:
            reason = str(decision.get("reason", ""))
        metadata["judge"] = {
            "model": MODEL_CROP_JUDGE,
            "choice": choice,
            "confidence": confidence,
            "reason": reason,
        }
    except Exception as exc:
        choice, reason = _fallback_crop_choice(full_lines, crop_lines)
        metadata["judge"] = {
            "model": "local_fallback",
            "choice": choice,
            "confidence": 0.0,
            "reason": reason,
            "error": str(exc),
        }
    metadata["selected"] = choice
    metadata["reason"] = metadata["judge"]["reason"]
    return (crop_lines if choice == "crops" else full_lines), metadata


def recheck(crop: Image.Image, block: dict) -> dict:
    prompt = RECHECK_PROMPT.format(a=block["a"] or "(nothing)", b=block["b"] or "(nothing)")
    try:
        result = parse_json(chat(MODEL_RECHECK, prompt, [crop]))
        return {
            "supported": bool(result.get("supported")),
            "text": str(result.get("text", "")).strip(),
        }
    except Exception as exc:
        message = f"recheck failed: {exc}"
        print(message, file=sys.stderr)
        return {"supported": False, "text": "", "error": message}
