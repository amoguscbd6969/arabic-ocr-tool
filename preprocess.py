"""CPU document preprocessing and Arabic paragraph detection.

The implementation intentionally has a strict quality gate. A bad crop layout
is worse than a full-page OCR input, so callers must check ``quality.passed``
before using the returned regions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass
class ParagraphBox:
    x0: int
    y0: int
    x1: int
    y1: int
    ink_fraction: float

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class LayoutQuality:
    passed: bool
    reason: str
    boxes: int
    ink_capture: float
    box_coverage: float

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class PreprocessResult:
    image: Image.Image
    binary: np.ndarray
    metadata: dict


def _to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _to_pil(image: np.ndarray) -> Image.Image:
    if image.ndim == 2:
        return Image.fromarray(image).convert("RGB")
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def _order_points(points: np.ndarray) -> np.ndarray:
    points = points.astype("float32")
    ordered = np.zeros((4, 2), dtype="float32")
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).ravel()
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(differences)]
    ordered[3] = points[np.argmax(differences)]
    return ordered


def _perspective_crop(image: np.ndarray) -> tuple[np.ndarray, bool]:
    """Find a photographed paper boundary and apply a four-point transform."""
    height, width = image.shape[:2]
    scale = min(1.0, 1200 / max(height, width))
    preview = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 45, 140)
    edges = cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    )
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    preview_area = preview.shape[0] * preview.shape[1]
    quadrilateral = None
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:12]:
        if cv2.contourArea(contour) < preview_area * 0.22:
            break
        perimeter = cv2.arcLength(contour, True)
        approximation = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approximation) == 4 and cv2.isContourConvex(approximation):
            quadrilateral = approximation.reshape(4, 2).astype("float32") / scale
            break
    if quadrilateral is None:
        return image, False

    top_left, top_right, bottom_right, bottom_left = _order_points(quadrilateral)
    target_width = round(max(
        np.linalg.norm(bottom_right - bottom_left),
        np.linalg.norm(top_right - top_left),
    ))
    target_height = round(max(
        np.linalg.norm(top_right - bottom_right),
        np.linalg.norm(top_left - bottom_left),
    ))
    if target_width < width * 0.35 or target_height < height * 0.35:
        return image, False
    destination = np.array(
        [
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1],
        ],
        dtype="float32",
    )
    transform = cv2.getPerspectiveTransform(
        np.array([top_left, top_right, bottom_right, bottom_left]), destination
    )
    return cv2.warpPerspective(image, transform, (target_width, target_height)), True


def _binary_ink(gray: np.ndarray) -> np.ndarray:
    block_size = max(31, (min(gray.shape) // 18) | 1)
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        15,
    )


def _line_extents(ink: np.ndarray) -> list[tuple[int, int, int, int]]:
    height, width = ink.shape
    row_counts = (ink > 0).sum(axis=1)
    active = row_counts > max(3, width * 0.008)
    gap_limit = max(2, round(height * 0.004))
    bands = []
    start = last = None
    for y, enabled in enumerate(active):
        if enabled:
            if start is None:
                start = y
            last = y
        elif start is not None and y - last > gap_limit:
            region = ink[start:last + 1]
            columns = np.flatnonzero((region > 0).any(axis=0))
            if len(columns) and last - start >= 2:
                bands.append((int(columns[0]), start, int(columns[-1] + 1), last + 1))
            start = last = None
    return bands


def _orientation_score(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ink = _binary_ink(gray)
    bands = _line_extents(ink)
    if len(bands) < 3:
        return -10.0
    height, width = ink.shape
    widths = np.array([x1 - x0 for x0, _, x1, _ in bands], dtype=float)
    heights = np.array([y1 - y0 for _, y0, _, y1 in bands], dtype=float)
    lefts = np.array([x0 for x0, _, _, _ in bands], dtype=float)
    rights = np.array([x1 for _, _, x1, _ in bands], dtype=float)
    horizontality = float(np.median(widths / np.maximum(1.0, heights)))
    # Arabic paragraphs normally have a more stable right edge and a ragged
    # left edge. This also helps distinguish 0 from 180 degrees.
    rtl_alignment = float((np.std(lefts) - np.std(width - rights)) / max(1, width))
    useful_lines = min(len(bands), 40) / 40
    row_projection = (ink > 0).mean(axis=1)
    column_projection = (ink > 0).mean(axis=0)
    projection_score = float(np.log(
        (row_projection.var() + 1e-9) / (column_projection.var() + 1e-9)
    ))
    return float(
        np.log1p(horizontality)
        + 2.5 * rtl_alignment
        + useful_lines
        + 1.5 * projection_score
    )


def detect_cardinal_orientation(image: np.ndarray) -> tuple[int, float, dict]:
    rotations = {
        0: image,
        90: cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
        180: cv2.rotate(image, cv2.ROTATE_180),
        270: cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE),
    }
    scores = {degrees: _orientation_score(candidate) for degrees, candidate in rotations.items()}
    ranked = sorted(scores, key=scores.get, reverse=True)
    best, second = ranked[:2]
    confidence = float(max(0.0, min(1.0, (scores[best] - scores[second]) / 1.5)))
    return best, confidence, {str(key): round(value, 4) for key, value in scores.items()}


def _rotate_cardinal(image: np.ndarray, degrees: int) -> np.ndarray:
    if degrees == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if degrees == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def _deskew(image: np.ndarray) -> tuple[np.ndarray, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(40, image.shape[1] // 8),
        minLineLength=max(40, image.shape[1] // 5),
        maxLineGap=max(10, image.shape[1] // 50),
    )
    angles = []
    if lines is not None:
        for x0, y0, x1, y1 in np.asarray(lines).reshape(-1, 4):
            angle = np.degrees(np.arctan2(y1 - y0, x1 - x0))
            if -12 <= angle <= 12:
                angles.append(angle)
    if not angles:
        return image, 0.0
    angle = float(np.median(angles))
    if abs(angle) < 0.25:
        return image, angle
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    rotated = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return rotated, angle


def preprocess_document(
    image: Image.Image,
    target_width: int = 2000,
    apply_orientation: bool = True,
) -> PreprocessResult:
    """Perspective-correct, orient, deskew and enhance a document photograph."""
    original = _to_bgr(image)
    original_height, original_width = original.shape[:2]
    corrected, perspective_applied = _perspective_crop(original)

    orientation, orientation_confidence, orientation_scores = detect_cardinal_orientation(corrected)
    # Do not rotate on a weak heuristic. The metadata allows an optional vision
    # model to resolve ambiguous orientation upstream.
    applied_orientation = orientation if apply_orientation and orientation_confidence >= 0.10 else 0
    corrected = _rotate_cardinal(corrected, applied_orientation)
    corrected, skew_angle = _deskew(corrected)

    height, width = corrected.shape[:2]
    scale = max(1.0, target_width / max(1, width))
    scale = min(scale, 5.0)
    if scale > 1.02:
        corrected = cv2.resize(
            corrected,
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_CUBIC,
        )

    gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
    background_kernel = max(31, (min(gray.shape) // 12) | 1)
    background = cv2.GaussianBlur(gray, (background_kernel, background_kernel), 0)
    normalized = cv2.divide(gray, np.maximum(background, 1), scale=245)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(normalized)
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    enhanced = cv2.addWeighted(enhanced, 1.45, blurred, -0.45, 0)
    binary = _binary_ink(enhanced)

    metadata = {
        "original_width": original_width,
        "original_height": original_height,
        "output_width": int(enhanced.shape[1]),
        "output_height": int(enhanced.shape[0]),
        "upscale": round(float(scale), 4),
        "perspective_applied": perspective_applied,
        "orientation_suggested": orientation,
        "orientation_applied": applied_orientation,
        "orientation_confidence": round(orientation_confidence, 4),
        "orientation_scores": orientation_scores,
        "deskew_angle": round(skew_angle, 4),
        "blur_variance": round(float(cv2.Laplacian(enhanced, cv2.CV_64F).var()), 2),
        "contrast_std": round(float(enhanced.std()), 2),
        "low_resolution_input": original_width < 1200,
    }
    return PreprocessResult(_to_pil(enhanced), binary, metadata)


def _boxes_overlap_or_neighbor(left: list[int], right: list[int], width: int, height: int) -> bool:
    x_overlap = max(0, min(left[2], right[2]) - max(left[0], right[0]))
    minimum_width = max(1, min(left[2] - left[0], right[2] - right[0]))
    vertical_gap = max(0, max(left[1], right[1]) - min(left[3], right[3]))
    y_overlap = max(0, min(left[3], right[3]) - max(left[1], right[1]))
    return (
        (x_overlap / minimum_width >= 0.30 and vertical_gap <= height * 0.035)
        or (y_overlap > 0 and x_overlap > 0)
        or (
            vertical_gap <= height * 0.018
            and abs(left[2] - right[2]) <= width * 0.12
        )
    )


def _merge_boxes(boxes: list[list[int]], width: int, height: int) -> list[list[int]]:
    changed = True
    while changed:
        changed = False
        merged = []
        while boxes:
            current = boxes.pop(0)
            index = 0
            while index < len(boxes):
                candidate = boxes[index]
                if _boxes_overlap_or_neighbor(current, candidate, width, height):
                    current = [
                        min(current[0], candidate[0]),
                        min(current[1], candidate[1]),
                        max(current[2], candidate[2]),
                        max(current[3], candidate[3]),
                    ]
                    boxes.pop(index)
                    changed = True
                else:
                    index += 1
            merged.append(current)
        boxes = merged
    return boxes


def detect_arabic_paragraphs(
    result: PreprocessResult,
    max_regions: int = 16,
) -> tuple[list[ParagraphBox], LayoutQuality]:
    """Detect paragraph regions with CPU morphology and reject weak layouts."""
    ink = result.binary.copy()
    height, width = ink.shape

    # Remove long page borders/rules without erasing normal Arabic strokes.
    horizontal = cv2.morphologyEx(
        ink,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, width // 3), 1)),
    )
    vertical = cv2.morphologyEx(
        ink,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(40, height // 3))),
    )
    text_ink = cv2.subtract(ink, cv2.bitwise_or(horizontal, vertical))

    words = cv2.dilate(
        text_ink,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, width // 90), max(2, height // 700))),
        iterations=1,
    )
    regions = cv2.morphologyEx(
        words,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(7, width // 180), max(11, height // 45))),
        iterations=1,
    )
    contours, _ = cv2.findContours(regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    raw_boxes = []
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        # Keep compact headers/page labels, but reject isolated punctuation and
        # margin line numbers. Body fragments are merged below.
        if box_width < width * 0.055 or box_height < height * 0.015:
            continue
        if box_width * box_height < width * height * 0.0013:
            continue
        raw_boxes.append([x, y, x + box_width, y + box_height])
    merged = _merge_boxes(raw_boxes, width, height)

    boxes = []
    total_ink = max(1, int((text_ink > 0).sum()))
    captured_mask = np.zeros_like(text_ink)
    total_box_area = 0
    padding_x = max(8, round(width * 0.012))
    padding_y = max(8, round(height * 0.009))
    for x0, y0, x1, y1 in merged:
        x0, y0 = max(0, x0 - padding_x), max(0, y0 - padding_y)
        x1, y1 = min(width, x1 + padding_x), min(height, y1 + padding_y)
        area = max(1, (x1 - x0) * (y1 - y0))
        ink_pixels = int((text_ink[y0:y1, x0:x1] > 0).sum())
        ink_fraction = ink_pixels / area
        if ink_fraction < 0.008 or ink_fraction > 0.55:
            continue
        boxes.append(ParagraphBox(x0, y0, x1, y1, round(ink_fraction, 4)))
        captured_mask[y0:y1, x0:x1] = 255
        total_box_area += area

    # Reading order is top-to-bottom, then right-to-left for regions sharing a row.
    row_tolerance = max(15, round(height * 0.03))
    boxes.sort(key=lambda box: (round(box.y0 / row_tolerance), -box.x1))
    captured = int(((text_ink > 0) & (captured_mask > 0)).sum()) / total_ink
    coverage = total_box_area / max(1, width * height)

    reason = "ok"
    passed = True
    if not boxes:
        passed, reason = False, "no paragraph regions"
    elif len(boxes) > max_regions:
        passed, reason = False, f"too many regions ({len(boxes)} > {max_regions})"
    elif captured < 0.72:
        passed, reason = False, f"low ink capture ({captured:.1%})"
    elif coverage < 0.06:
        passed, reason = False, f"regions too sparse ({coverage:.1%})"
    elif coverage > 0.92:
        passed, reason = False, f"regions cover almost the full page ({coverage:.1%})"
    elif len(boxes) == 1 and coverage > 0.85:
        passed, reason = False, "single crop is equivalent to full-page OCR"

    quality = LayoutQuality(
        passed=passed,
        reason=reason,
        boxes=len(boxes),
        ink_capture=round(captured, 4),
        box_coverage=round(coverage, 4),
    )
    return boxes, quality


def crop_paragraphs(
    result: PreprocessResult,
    boxes: list[ParagraphBox],
    minimum_width: int = 1200,
) -> list[Image.Image]:
    crops = []
    for box in boxes:
        crop = result.image.crop((box.x0, box.y0, box.x1, box.y1))
        if crop.width < minimum_width:
            scale = min(3.0, minimum_width / max(1, crop.width))
            crop = crop.resize(
                (round(crop.width * scale), round(crop.height * scale)),
                Image.Resampling.LANCZOS,
            )
        crops.append(crop)
    return crops
