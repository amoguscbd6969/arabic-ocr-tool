"""ArabicOCR_KHATT adapter used exclusively by the ``--free`` profile.

The upstream package owns page segmentation, line preprocessing, CRNN inference,
and CTC decoding. The model is loaded lazily once per command and reused for
every image in a batch.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import asdict, dataclass

import cv2
import numpy as np
from PIL import Image


MODEL_NAME = "FixFips/arabicocr-khatt (CRNN-CTC)"
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
    global _OCR, _LOAD_ERROR
    if _OCR is not None:
        return _OCR
    if _LOAD_ERROR is not None:
        raise RuntimeError(str(_LOAD_ERROR)) from _LOAD_ERROR
    try:
        from arabicocr_khatt import ArabicOCR

        device = os.getenv("ARABICOCR_KHATT_DEVICE") or None
        _OCR = ArabicOCR.from_pretrained(device=device)
        return _OCR
    except Exception as exc:
        _LOAD_ERROR = exc
        raise RuntimeError(
            "ArabicOCR_KHATT is unavailable. Install it with "
            "`python -m pip install arabicocr-khatt`. The first run also needs "
            f"network access to download its checkpoint. Original error: {exc}"
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


def recognize(img: Image.Image) -> LocalOCRResult:
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
            MODEL_NAME,
            lines,
            time.perf_counter() - started,
            segmentation="tiled_line_segmentation" if tiled else "upstream_page_segmentation",
        )
    except Exception as exc:
        return LocalOCRResult(MODEL_NAME, [], time.perf_counter() - started, str(exc))
