"""Arabic OCR command entry point and pipeline orchestrator.

Remote paid OCR is isolated in ``paid_ocr.py``. This module resolves inputs,
runs local preprocessing, coordinates OCR stages, and writes managed artifacts.

Setup: pip install requests pillow python-dotenv numpy opencv-python-headless
Run:   python main.py sample.jpg
       python main.py report.pdf --document-mode mixed
       python main.py report.pdf --document-mode full-ocr

Managed layout:
- input/<source files>
- log/<source name>/<preprocessing, model transcripts, reports>
- output/<source name>/final.txt
"""

import argparse
import difflib
import json
import random
import re
import sys
import unicodedata
from pathlib import Path

from PIL import Image, ImageOps

import calculate_cost
import convert
import paid_ocr
import preprocess

PROJECT_DIR = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_DIR / "input"
LOG_DIR = PROJECT_DIR / "log"
OUTPUT_DIR = PROJECT_DIR / "output"

DOCUMENT_MODES = {"mixed", "full-ocr"}

FUZZY_AGREEMENT = 0.85
ALIGNMENT_GAP_COST = 0.45
AUDIT_RATE = 0.05
IGNORE_DIACRITICS = False
UNCLEAR_RE = re.compile(r"[\[\(\{]\s*غير\s+واضح\s*[\]\)\}]")
def resolve_input_path(path: str | Path) -> Path:
    """Resolve explicit paths first, then convenient names inside input/."""
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    managed_candidate = INPUT_DIR / candidate
    if managed_candidate.is_file():
        return managed_candidate
    raise FileNotFoundError(candidate)


def sanitize_artifact_name(name: str) -> str:
    """Remove invisible, unsafe, and Windows-reserved filename characters."""
    clean = re.sub(
        r"[\x00-\x1f\x7f-\x9f\u200e\u200f\u202a-\u202e\ufeff]", "", name
    ).strip()
    clean = re.sub(r'[\\/*?:"<>| ]', "_", clean)
    return clean or "document"


def artifact_directories(path: str | Path) -> tuple[Path, Path]:
    """Return and create the managed log/output folders for one input."""
    name = sanitize_artifact_name(Path(path).stem)
    log_dir = (LOG_DIR / name).resolve()
    output_dir = (OUTPUT_DIR / name).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return log_dir, output_dir


def source_signature(path: str | Path) -> dict:
    """Return a cheap source fingerprint used to reject stale OCR output."""
    stat = Path(path).stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def find_cached_ocr_output(
    path: str | Path,
    document_mode: str = "mixed",
) -> Path | None:
    """Return a reusable managed OCR result, or None when it may be stale."""
    input_path = resolve_input_path(path)
    artifact_name = sanitize_artifact_name(input_path.stem)
    log_dir = LOG_DIR / artifact_name
    output_path = OUTPUT_DIR / artifact_name / "final.txt"
    if not output_path.is_file() or output_path.stat().st_size == 0:
        return None

    if input_path.suffix.lower() in convert.IMAGE_EXTENSIONS:
        report_candidates = [log_dir / f"{artifact_name}.final.json"]
    else:
        report_candidates = [
            log_dir / document_mode / f"{artifact_name}.final.json",
        ]
        # Compatibility with document reports created before mode-specific log
        # directories were introduced. Those reports represent mixed mode.
        if document_mode == "mixed":
            report_candidates.append(log_dir / f"{artifact_name}.final.json")

    report_path = next((candidate for candidate in report_candidates if candidate.is_file()), None)
    if (
        input_path.suffix.lower() not in convert.IMAGE_EXTENSIONS
        and document_mode != "mixed"
        and report_path is None
    ):
        return None
    if report_path is not None:
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if input_path.suffix.lower() not in convert.IMAGE_EXTENSIONS:
            recorded_mode = report.get("document_mode", "mixed")
            if recorded_mode != document_mode:
                return None
        recorded_signature = report.get("source_signature")
        if recorded_signature is not None:
            return output_path if recorded_signature == source_signature(input_path) else None

    # Backward-compatible check for older reports without a signature. A source
    # modified after the OCR output is treated as stale and must be processed.
    if output_path.stat().st_mtime_ns < input_path.stat().st_mtime_ns:
        return None
    return output_path


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("\u0640", "")
    if IGNORE_DIACRITICS:
        text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def contains_abstention(text: str) -> bool:
    return bool(UNCLEAR_RE.search(text))


def similarity(left: str, right: str) -> float:
    return difflib.SequenceMatcher(None, norm(left), norm(right), autojunk=False).ratio()


def preprocess_for_ocr(img: Image.Image) -> preprocess.PreprocessResult:
    """Run local document cleanup without any orientation correction."""
    return preprocess.preprocess_document(img, apply_orientation=False)


def segment_line_bands(img: Image.Image) -> list[tuple[int, int]]:
    """Find visual text-line bands from pixels, not model-reported coordinates."""
    original_width, original_height = img.size
    scale = min(1.0, 1600 / max(1, original_width))
    width = max(1, round(original_width * scale))
    height = max(1, round(original_height * scale))
    gray = ImageOps.autocontrast(img.convert("L").resize((width, height)))
    pixels = gray.load()
    row_ink_counts = []
    for y in range(height):
        dark_pixels = sum(1 for x in range(width) if pixels[x, y] < 170)
        row_ink_counts.append(dark_pixels)

    # A fixed threshold fails badly on noisy scans: almost every row contains
    # a few dark scanner artifacts. The lower row-count percentile estimates
    # that noise floor while retaining clean-page behavior.
    sorted_counts = sorted(row_ink_counts)
    noise_floor = sorted_counts[int((len(sorted_counts) - 1) * 0.30)]
    minimum_ink = max(3, round(width * 0.005), noise_floor)
    active_rows = [count >= minimum_ink for count in row_ink_counts]

    raw_bands = []
    start = None
    for y, active in enumerate(active_rows + [False]):
        if active and start is None:
            start = y
        elif not active and start is not None:
            if y - start >= 2:
                raw_bands.append([start, y])
            start = None

    join_gap = max(2, round(height * 0.0025))
    joined = []
    for top, bottom in raw_bands:
        if joined and top - joined[-1][1] <= join_gap:
            joined[-1][1] = bottom
        else:
            joined.append([top, bottom])
    y_scale = original_height / height
    return [
        (max(0, round(top * y_scale)), min(original_height, round(bottom * y_scale)))
        for top, bottom in joined
    ]


def fit_bands_to_line_count(
    bands: list[tuple[int, int]], expected_count: int
) -> list[tuple[int, int]]:
    """Merge over-segmented components until they match the OCR line count."""
    fitted = [list(band) for band in bands]
    while expected_count > 0 and len(fitted) > expected_count:
        merge_at = min(
            range(len(fitted) - 1),
            key=lambda index: fitted[index + 1][0] - fitted[index][1],
        )
        fitted[merge_at][1] = fitted[merge_at + 1][1]
        del fitted[merge_at + 1]
    return [tuple(band) for band in fitted]


def crop_visual_line(
    img: Image.Image,
    bands: list[tuple[int, int]],
    ordinal: int,
    total_blocks: int,
) -> Image.Image:
    if not bands:
        return img
    index = 0 if total_blocks <= 1 else round(ordinal / (total_blocks - 1) * (len(bands) - 1))
    top, bottom = bands[max(0, min(len(bands) - 1, index))]
    padding = max(12, round(img.height * 0.008))
    return img.crop((0, max(0, top - padding), img.width, min(img.height, bottom + padding)))


def align(a: list[str], b: list[str]) -> list[dict]:
    """Globally align OCR lines and label fuzzy agreement or abstention."""
    rows, columns = len(a) + 1, len(b) + 1
    costs = [[0.0] * columns for _ in range(rows)]
    moves = [[""] * columns for _ in range(rows)]
    for i in range(1, rows):
        costs[i][0], moves[i][0] = i * ALIGNMENT_GAP_COST, "a"
    for j in range(1, columns):
        costs[0][j], moves[0][j] = j * ALIGNMENT_GAP_COST, "b"
    for i in range(1, rows):
        for j in range(1, columns):
            options = {
                "pair": costs[i - 1][j - 1] + 1.0 - similarity(a[i - 1], b[j - 1]),
                "a": costs[i - 1][j] + ALIGNMENT_GAP_COST,
                "b": costs[i][j - 1] + ALIGNMENT_GAP_COST,
            }
            move = min(options, key=options.get)
            costs[i][j], moves[i][j] = options[move], move

    aligned = []
    i, j = len(a), len(b)
    while i or j:
        move = moves[i][j]
        if move == "pair":
            left, right = a[i - 1], b[j - 1]
            score = similarity(left, right)
            left_abstains, right_abstains = contains_abstention(left), contains_abstention(right)
            if left_abstains != right_abstains:
                status = "abstention"
            elif not left_abstains and score >= FUZZY_AGREEMENT:
                status = "agree"
            else:
                status = "dispute"
            aligned.append({
                "status": status, "a": left, "b": right,
                "similarity": round(score, 4), "a_index": i - 1, "b_index": j - 1,
            })
            i -= 1
            j -= 1
        elif move == "a":
            aligned.append({
                "status": "single_source", "a": a[i - 1], "b": "", "similarity": 0.0,
                "a_index": i - 1, "b_index": None,
            })
            i -= 1
        else:
            aligned.append({
                "status": "single_source", "a": "", "b": b[j - 1], "similarity": 0.0,
                "a_index": None, "b_index": j - 1,
            })
            j -= 1
    aligned.reverse()
    return aligned


def process(
    path: str,
    *,
    log_dir: Path | None = None,
    output_path: Path | None = None,
    artifact_name: str | None = None,
) -> Path:
    image_path = Path(path)
    raw_img = Image.open(image_path).convert("RGB")
    artifact_name = sanitize_artifact_name(artifact_name or image_path.stem)

    if log_dir is None:
        log_dir, default_output_dir = artifact_directories(image_path)
    else:
        log_dir = Path(log_dir).resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        default_output_dir = (OUTPUT_DIR / artifact_name).resolve()
        default_output_dir.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        default_output_dir.mkdir(parents=True, exist_ok=True)
        output_path = default_output_dir / "final.txt"
    else:
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

    prepared = preprocess_for_ocr(raw_img)
    img = prepared.image

    # Đảm bảo đường dẫn tuyệt đối, chuẩn chuỗi string tránh lỗi Errno 22
    preprocessed_path = (log_dir / f"{artifact_name}_preprocessed.png").resolve()
    img.save(str(preprocessed_path), format="PNG")

    paragraph_boxes, layout_quality = preprocess.detect_arabic_paragraphs(prepared)

    if paid_ocr.ACTIVE_MODEL_PROFILE == "free":
        import local_ocr

        # ArabicOCR_KHATT includes preprocessing calibrated for KHATT. Feeding
        # it our binarized preview can erase strokes, so recognize the raw page.
        local_result = local_ocr.recognize(raw_img)
        if local_result.error or not local_result.lines:
            raise RuntimeError(
                f"{local_ocr.MODEL_NAME} failed: "
                f"{local_result.error or 'no text returned'}"
            )
        final = [
            {"text": line, "flag": "single_local_model", "similarity": 1.0}
            for line in local_result.lines
        ]
        (log_dir / f"{artifact_name}_modelA.txt").write_text(
            "\n".join(local_result.lines), encoding="utf-8"
        )
        # Prevent artifacts from an older dual/evolutionary run looking current.
        for stale_suffix in ("_modelB.txt", "_draft.txt", "_retry.txt", ".mark.json"):
            (log_dir / f"{artifact_name}{stale_suffix}").unlink(missing_ok=True)
        output_path.write_text(
            "\n".join(item["text"] for item in final), encoding="utf-8"
        )
        report = {
            "input": str(image_path),
            "source_signature": source_signature(image_path),
            "model_profile": paid_ocr.ACTIVE_MODEL_PROFILE,
            "document_type": "unclassified",
            "classification_confidence": None,
            "model_domain": "Arabic handwriting (KHATT)",
            "models": {
                "primary": local_ocr.MODEL_NAME,
                "secondary": None,
                "recheck": None,
            },
            "local_ocr": local_result.as_dict(),
            "preprocessing": prepared.metadata,
            "preprocessed_image": str(preprocessed_path),
            "paragraph_detection": {
                "quality": layout_quality.as_dict(),
                "boxes": [box.as_dict() for box in paragraph_boxes],
            },
            "crop_route": {
                "attempted": False,
                "selected": local_result.segmentation,
                "reason": "local heuristic segmentation followed by KHATT CRNN-CTC only",
            },
            "detected_line_bands": len(local_result.lines),
            "lines": final,
        }
        (log_dir / f"{artifact_name}.final.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"{path}: classification=skipped; primary={local_ocr.MODEL_NAME}; "
            f"{len(final)} lines; {local_result.elapsed_seconds:.2f}s"
        )
        return output_path

    classification = paid_ocr.classify_document(img)
    primary_model = paid_ocr.select_primary_model(classification["type"])
    print(
        f"{path}: {classification['type']} "
        f"(confidence {classification['confidence']:.0%}); primary={primary_model}"
    )
    if classification["type"] == "printed":
        a, crop_route = paid_ocr.printed_ocr_with_optional_crops(
            img, prepared, paragraph_boxes, layout_quality
        )
    else:
        a = paid_ocr.ocr(primary_model, img)
        crop_route = {
            "attempted": False,
            "selected": "full",
            "reason": f"crop OCR disabled for {classification['type']} page",
        }
    b = paid_ocr.ocr(paid_ocr.MODEL_SECONDARY, img)
    (log_dir / f"{artifact_name}_modelA.txt").write_text("\n".join(a), encoding="utf-8")
    (log_dir / f"{artifact_name}_modelB.txt").write_text("\n".join(b), encoding="utf-8")

    blocks = align(a, b)
    raw_bands = segment_line_bands(img)
    bands = fit_bands_to_line_count(raw_bands, max(len(a), len(b), len(blocks)))
    final = []
    for ordinal, block in enumerate(blocks):
        base = {"similarity": block["similarity"]}
        if block["status"] == "agree":
            if random.random() < AUDIT_RATE:
                result = paid_ocr.recheck(
                    crop_visual_line(img, bands, ordinal, len(blocks)), block
                )
                if result["supported"] and norm(result["text"]) != norm(block["a"]):
                    final.append({"text": result["text"], "flag": "corrected_by_audit", **base})
                    continue
            final.append({"text": block["a"], "flag": "agreed", **base})
            continue

        if block["status"] == "abstention":
            if contains_abstention(block["a"]):
                text, flag = block["b"], "single_source_b"
            else:
                text, flag = block["a"], "single_source_a"
            final.append({"text": text, "flag": flag, **base})
            continue

        if block["status"] == "single_source":
            text = block["a"] or block["b"]
            flag = "single_source_a" if block["a"] else "single_source_b"
            final.append({"text": text, "flag": flag, **base})
            continue

        result = paid_ocr.recheck(
            crop_visual_line(img, bands, ordinal, len(blocks)), block
        )
        if result["supported"] and result["text"]:
            final.append({"text": result["text"], "flag": "corrected", **base})
        else:
            unresolved = {
                "text": f"[[UNRESOLVED | A: {block['a']} | B: {block['b']}]]",
                "flag": "unresolved", **base,
            }
            if result.get("error"):
                unresolved["error"] = result["error"]
            final.append(unresolved)

    output_path.write_text("\n".join(item["text"] for item in final), encoding="utf-8")
    output = {
        "input": str(image_path),
        "source_signature": source_signature(image_path),
        "model_profile": paid_ocr.ACTIVE_MODEL_PROFILE,
        "document_type": classification["type"],
        "classification_confidence": classification["confidence"],
        "models": {
            "primary": primary_model,
            "secondary": paid_ocr.MODEL_SECONDARY,
            "recheck": paid_ocr.MODEL_RECHECK,
        },
        "preprocessing": prepared.metadata,
        "preprocessed_image": str(preprocessed_path),
        "paragraph_detection": {
            "quality": layout_quality.as_dict(),
            "boxes": [box.as_dict() for box in paragraph_boxes],
        },
        "crop_route": crop_route,
        "detected_line_bands": len(raw_bands),
        "lines": final,
    }
    (log_dir / f"{artifact_name}.final.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    unresolved_count = sum(item["flag"] == "unresolved" for item in final)
    print(f"{path}: {len(final)} lines, {unresolved_count} unresolved")
    return output_path


def evolve(
    path: str,
    candidate_path: Path,
    *,
    log_dir: Path | None = None,
    artifact_name: str | None = None,
) -> Path:
    """Run prompt-diverse retries and make the adjudicated text authoritative."""
    # When this file is executed as __main__, let mark.py reuse this exact module
    # instance instead of importing a second copy with separate usage counters.
    sys.modules.setdefault("main", sys.modules[__name__])
    import mark

    mark.configure_models(
        paid_ocr.MODEL_MARK_BLIND,
        paid_ocr.MODEL_MARK_FORENSIC,
        paid_ocr.MODEL_MARK_JUDGE,
    )

    image_path = Path(path)
    artifact_name = sanitize_artifact_name(artifact_name or image_path.stem)
    if log_dir is None:
        log_dir, _ = artifact_directories(image_path)
    else:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
    preprocessed_path = log_dir / f"{artifact_name}_preprocessed.png"
    evaluation_image = preprocessed_path if preprocessed_path.exists() else image_path
    report = mark.run(evaluation_image, candidate_path)
    report["original_image"] = str(image_path)
    report["evaluation_image"] = str(evaluation_image)

    # Preserve both variants and let the evaluator select the authoritative one.
    # If evaluation fails, these writes never run, so the baseline stays usable.
    draft_path = log_dir / f"{artifact_name}_draft.txt"
    draft_path.write_text(candidate_path.read_text(encoding="utf-8"), encoding="utf-8")
    retry_path = log_dir / f"{artifact_name}_retry.txt"
    retry_text = "\n".join(report["final_lines"])
    retry_path.write_text(retry_text, encoding="utf-8")

    score_better_or_equal = report["final_score_10"] >= report["current_score_10"]
    wer_better_or_equal = report["final_estimated_wer"] <= report["current_estimated_wer"]
    strictly_better = (
        report["final_score_10"] > report["current_score_10"]
        or report["final_estimated_wer"] < report["current_estimated_wer"]
    )
    accepted = score_better_or_equal and wer_better_or_equal and strictly_better
    report["selection"] = {
        "accepted_retry": accepted,
        "authoritative_output": str(candidate_path),
        "rule": "score must not decrease, estimated WER must not increase, and one must improve",
    }
    if accepted:
        candidate_path.write_text(retry_text, encoding="utf-8")

    report_path = log_dir / f"{artifact_name}.mark.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"{path}: evolved {report['current_score_10']:.1f}/10 -> "
        f"{report['final_score_10']:.1f}/10; estimated WER "
        f"{report['current_estimated_wer']:.1%} -> {report['final_estimated_wer']:.1%}"
    )
    decision = "accepted retry" if accepted else "kept baseline"
    print(
        f"Selection: {decision}; final: {candidate_path}; draft: {draft_path}; "
        f"retry: {retry_path}; report: {report_path}"
    )
    return candidate_path


def _ocr_annotation(source: str, document_mode: str) -> str:
    if document_mode == "full-ocr":
        return f"[[OCR_PAGE | {source}]]"
    if re.fullmatch(r"page \d+", source):
        return f"[[OCR_PAGE_SCAN | {source}]]"
    return f"[[OCR_IMAGE | {source}]]"


def process_document(
    path: str,
    use_evolution: bool = True,
    document_mode: str = "mixed",
) -> Path:
    """Process documents in mixed extraction mode or full-page OCR mode."""
    if document_mode not in DOCUMENT_MODES:
        raise ValueError(f"document_mode must be one of {sorted(DOCUMENT_MODES)}")
    input_path = Path(path)
    artifact_name = sanitize_artifact_name(input_path.stem)
    document_log_dir, output_dir = artifact_directories(input_path)
    log_dir = document_log_dir / document_mode
    log_dir.mkdir(parents=True, exist_ok=True)
    parts = (
        convert.render_document_pages(input_path)
        if document_mode == "full-ocr"
        else convert.load_document(input_path)
    )
    if not parts:
        raise ValueError(f"No readable text or OCR-worthy images found in {input_path}")

    final_lines = []
    draft_lines = []
    part_reports = []
    for ordinal, part in enumerate(parts, start=1):
        summary = part.summary()
        if part.kind == "text":
            lines = [line.strip() for line in str(part.content).splitlines() if line.strip()]
            final_lines.extend(lines)
            draft_lines.extend(lines)
            summary.update({"method": "direct_text_extraction", "output_lines": len(lines)})
            part_reports.append(summary)
            print(f"{path}: {part.source}: extracted {len(lines)} typed lines directly")
            continue

        part_name = f"part_{ordinal:04d}"
        part_log_dir = log_dir / "parts" / part_name
        part_log_dir.mkdir(parents=True, exist_ok=True)
        source_image = part_log_dir / f"{part_name}_source.png"
        part.content.save(source_image, format="PNG")
        candidate_path = process(
            str(source_image),
            log_dir=part_log_dir,
            output_path=part_log_dir / f"{part_name}_ocr.txt",
            artifact_name=part_name,
        )
        baseline_lines = [
            line.strip()
            for line in candidate_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        annotation = _ocr_annotation(part.source, document_mode)
        draft_lines.append(annotation)
        draft_lines.extend(baseline_lines)
        if use_evolution:
            evolve(
                str(source_image),
                candidate_path,
                log_dir=part_log_dir,
                artifact_name=part_name,
            )
        selected_lines = [
            line.strip()
            for line in candidate_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        final_lines.append(annotation)
        final_lines.extend(selected_lines)
        summary.update({
            "method": "ocr_with_evolution" if use_evolution else "ocr_baseline",
            "draft_lines": len(baseline_lines),
            "output_lines": len(selected_lines),
            "output_annotation": annotation,
            "log_directory": str(part_log_dir),
        })
        mark_report = part_log_dir / f"{part_name}.mark.json"
        if mark_report.exists():
            evaluation = json.loads(mark_report.read_text(encoding="utf-8"))
            summary["evaluation"] = {
                "current_score_10": evaluation.get("current_score_10"),
                "final_score_10": evaluation.get("final_score_10"),
                "current_estimated_wer": evaluation.get("current_estimated_wer"),
                "final_estimated_wer": evaluation.get("final_estimated_wer"),
                "accepted_retry": evaluation.get("selection", {}).get("accepted_retry"),
            }
        part_reports.append(summary)

    output_path = output_dir / "final.txt"
    output_path.write_text("\n".join(final_lines), encoding="utf-8")
    if use_evolution and any(part.kind == "image" for part in parts):
        (log_dir / f"{artifact_name}_draft.txt").write_text(
            "\n".join(draft_lines), encoding="utf-8"
        )
    report = {
        "input": str(input_path),
        "input_type": input_path.suffix.lower(),
        "source_signature": source_signature(input_path),
        "model_profile": paid_ocr.ACTIVE_MODEL_PROFILE,
        "document_mode": document_mode,
        "parts": part_reports,
        "typed_parts": sum(part.kind == "text" for part in parts),
        "image_parts": sum(part.kind == "image" for part in parts),
        "output_lines": len(final_lines),
        "output": str(output_path),
    }
    (log_dir / f"{artifact_name}.final.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"{path}: complete; {report['typed_parts']} typed parts, "
        f"{report['image_parts']} OCR parts -> {output_path}"
    )
    return output_path


def process_input(
    path: str,
    use_evolution: bool = True,
    document_mode: str = "mixed",
) -> Path:
    """Dispatch a regular image or a supported document through the pipeline."""
    input_path = resolve_input_path(path)
    if paid_ocr.ACTIVE_MODEL_PROFILE == "free" and use_evolution:
        print(
            f"{input_path}: evolutionary retry disabled for --free "
            "(single local CRNN-CTC model)"
        )
        use_evolution = False
    extension = input_path.suffix.lower()
    if extension not in convert.SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(convert.SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported input type {extension!r}. Supported: {supported}")
    if extension in convert.IMAGE_EXTENSIONS:
        candidate = process(str(input_path))
        return evolve(str(input_path), candidate) if use_evolution else candidate
    return process_document(
        str(input_path),
        use_evolution=use_evolution,
        document_mode=document_mode,
    )


def translate_output(
    input_path: str | Path,
    final_text_path: str | Path,
    target_language: str,
    translation_mode: str = "verified",
    translation_profile: str = "free",
) -> Path:
    """Translate a completed OCR result into one managed output artifact."""
    import translate

    translate.configure_model_profile(translation_profile)

    resolved_input = resolve_input_path(input_path)
    log_dir, output_dir = artifact_directories(resolved_input)
    target_name, target_code = translate.normalize_target(target_language)
    translation_log = (
        log_dir
        / "translation"
        / target_code
        / translation_mode
        / translation_profile
    )
    translation_output = output_dir / "translations" / f"{target_code}.txt"
    result = translate.translate_file(
        final_text_path,
        target_code,
        mode=translation_mode,
        output_path=translation_output,
        log_dir=translation_log,
    )
    print(
        f"{resolved_input}: translated to {target_name} ({translation_mode}) -> {result}"
    )
    return result


def preprocess_only(path: str) -> Path:
    """Write an OCR-ready image and metadata without making any API calls."""
    input_path = resolve_input_path(path)
    if input_path.suffix.lower() not in convert.IMAGE_EXTENSIONS:
        raise ValueError("--preprocess-only currently accepts image files only")
    with Image.open(input_path) as opened:
        result = preprocess_for_ocr(opened.convert("RGB"))
    boxes, quality = preprocess.detect_arabic_paragraphs(result)
    artifact_name = sanitize_artifact_name(input_path.stem)
    log_dir, _ = artifact_directories(input_path)
    output_path = log_dir / f"{artifact_name}_preprocessed.png"
    report_path = log_dir / f"{artifact_name}.preprocess.json"
    result.image.save(output_path, format="PNG", optimize=True)
    report_path.write_text(
        json.dumps(
            {
                "input": str(input_path),
                "output": str(output_path),
                "model_profile": paid_ocr.ACTIVE_MODEL_PROFILE,
                "preprocessing": result.metadata,
                "paragraph_detection": {
                    "quality": quality.as_dict(),
                    "boxes": [box.as_dict() for box in boxes],
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Preprocessed image: {output_path}")
    print(f"Preprocess report: {report_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Adaptive Arabic OCR for images, PDF, Word, PowerPoint, spreadsheets, and text files."
    )
    parser.add_argument("inputs", nargs="+", help="one or more image or document paths")
    parser.add_argument(
        "--no-evolve",
        action="store_true",
        help="skip the two retry prompts and profile-selected visual adjudication",
    )
    parser.add_argument(
        "--preprocess-only",
        action="store_true",
        help="write enhanced image and preprocessing report, then stop before OCR/API calls",
    )
    parser.add_argument(
        "--document-mode",
        choices=sorted(DOCUMENT_MODES),
        default="mixed",
        help=(
            "mixed: extract typed text directly and OCR only images/scans; "
            "full-ocr: render and OCR every page"
        ),
    )
    parser.add_argument(
        "--translate",
        action="append",
        metavar="LANG",
        help="translate final OCR text; repeat for multiple targets, e.g. --translate en --translate vi",
    )
    parser.add_argument(
        "--translation-mode",
        choices=["verified", "fast"],
        default="verified",
        help=(
            "verified: local direct/pivot + free multilingual + profile-selected judge; "
            "fast: local direct/pivot only for configured targets "
            "(zero OpenRouter API calls)"
        ),
    )
    parser.add_argument(
        "--translation-profile",
        choices=["free", "lite", "high", "max"],
        default="free",
        help=(
            "model profile used only for verified translation judging "
            "(default: free; independent of the OCR profile)"
        ),
    )
    parser.add_argument(
        "--force-ocr",
        action="store_true",
        help="ignore an existing output/<name>/final.txt and run OCR again before translation",
    )
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
    paid_ocr.configure_model_profile(args.model_profile)
    print(f"Model profile: {paid_ocr.ACTIVE_MODEL_PROFILE}")
    if args.preprocess_only and args.translate:
        parser.error("--translate cannot be combined with --preprocess-only")
    for input_path in args.inputs:
        if args.preprocess_only:
            preprocess_only(input_path)
        else:
            final_text_path = None
            if args.translate and not args.force_ocr:
                final_text_path = find_cached_ocr_output(
                    input_path,
                    document_mode=args.document_mode,
                )
                if final_text_path is not None:
                    print(f"{input_path}: using existing OCR output -> {final_text_path}")
            if final_text_path is None:
                final_text_path = process_input(
                    input_path,
                    use_evolution=not args.no_evolve,
                    document_mode=args.document_mode,
                )
            for target_language in args.translate or []:
                translate_output(
                    input_path,
                    final_text_path,
                    target_language,
                    translation_mode=args.translation_mode,
                    translation_profile=args.translation_profile,
                )
    translation_module = sys.modules.get("translate")
    translation_usage = (
        getattr(translation_module, "MODEL_USAGE", {})
        if translation_module is not None
        else {}
    )
    cost_report = calculate_cost.build_cost_report(
        {"ocr": paid_ocr.MODEL_USAGE, "translation": translation_usage},
        command=sys.argv,
    )
    calculate_cost.print_cost_report(cost_report)
