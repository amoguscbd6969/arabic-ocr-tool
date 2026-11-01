"""Preprocess the KHATT training archive into a fine-tuning-ready dataset.

The source archive contains JPEG images while its CSV manifests still refer to
the original TIFF filenames. This utility preserves every label token, writes
enhanced JPEGs, and updates only the manifest filename column.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

import cv2
from PIL import Image

import preprocess


SPLITS = {
    "Train_deskewed/Train_deskewed/": ("Train", "Train.csv"),
    "Validate_deskewed/Validate_deskewed/": ("Validation", "Validation.csv"),
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

_ARCHIVE: zipfile.ZipFile | None = None
_OUTPUT_ROOT: Path | None = None
_TARGET_WIDTH = 2000
_FORCE = False


def _initialize_worker(
    archive_path: str,
    output_root: str,
    target_width: int,
    force: bool,
) -> None:
    global _ARCHIVE, _OUTPUT_ROOT, _TARGET_WIDTH, _FORCE
    cv2.setNumThreads(1)
    _ARCHIVE = zipfile.ZipFile(archive_path)
    _OUTPUT_ROOT = Path(output_root)
    _TARGET_WIDTH = target_width
    _FORCE = force


def _split_for_member(member: str) -> str:
    for prefix, (split, _) in SPLITS.items():
        if member.startswith(prefix):
            return split
    raise ValueError(f"Archive member is outside a known split: {member}")


def _process_member(member: str) -> dict:
    if _ARCHIVE is None or _OUTPUT_ROOT is None:
        raise RuntimeError("Dataset worker was not initialized")

    split = _split_for_member(member)
    source_name = PurePosixPath(member).name
    output_name = f"{Path(source_name).stem}.jpg"
    output_path = _OUTPUT_ROOT / split / output_name
    if output_path.exists() and not _FORCE:
        return {
            "source": member,
            "output": output_path.relative_to(_OUTPUT_ROOT).as_posix(),
            "split": split,
            "status": "skipped",
        }

    with _ARCHIVE.open(member) as source:
        with Image.open(source) as opened:
            image = opened.convert("RGB")
            image.load()

    # Match main.py: orientation is recorded by preprocess.py but never applied.
    result = preprocess.preprocess_document(
        image,
        target_width=_TARGET_WIDTH,
        apply_orientation=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp")
    result.image.save(temporary_path, format="JPEG", quality=95, optimize=True)
    temporary_path.replace(output_path)
    return {
        "source": member,
        "output": output_path.relative_to(_OUTPUT_ROOT).as_posix(),
        "split": split,
        "status": "processed",
        "preprocessing": result.metadata,
    }


def _image_members(archive: zipfile.ZipFile) -> list[str]:
    members = []
    for info in archive.infolist():
        if info.is_dir():
            continue
        suffix = PurePosixPath(info.filename).suffix.lower()
        if suffix in IMAGE_SUFFIXES and any(
            info.filename.startswith(prefix) for prefix in SPLITS
        ):
            members.append(info.filename)
    return sorted(members)


def _rewrite_manifest(
    archive: zipfile.ZipFile,
    source_name: str,
    destination: Path,
) -> tuple[int, list[str]]:
    with archive.open(source_name) as raw:
        reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty manifest: {source_name}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    stems = []
    with destination.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(rows[0])
        for row in rows[1:]:
            if not row:
                continue
            stem = Path(row[0]).stem
            stems.append(stem)
            row[0] = f"{stem}.jpg"
            writer.writerow(row)
    return len(stems), stems


def _create_archive(source_root: Path, archive_path: Path) -> None:
    temporary_path = archive_path.with_suffix(".tmp")
    if temporary_path.exists():
        temporary_path.unlink()
    with zipfile.ZipFile(
        temporary_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as bundle:
        for path in sorted(source_root.rglob("*")):
            if path.is_file():
                archive_name = Path(source_root.name) / path.relative_to(source_root)
                bundle.write(path, archive_name.as_posix())
    temporary_path.replace(archive_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess KHATT data.zip for OCR fine-tuning."
    )
    parser.add_argument("archive", nargs="?", type=Path, default=Path("data.zip"))
    parser.add_argument("--output", type=Path, default=Path("data_preprocessed"))
    parser.add_argument("--target-width", type=int, default=2000)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(4, os.cpu_count() or 1)),
    )
    parser.add_argument("--force", action="store_true", help="reprocess existing images")
    parser.add_argument(
        "--no-archive",
        action="store_true",
        help="leave only the output directory and do not create a ZIP",
    )
    args = parser.parse_args()

    archive_path = args.archive.resolve()
    output_root = args.output.resolve()
    if not archive_path.is_file():
        parser.error(f"archive not found: {archive_path}")
    if args.target_width <= 0:
        parser.error("--target-width must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")

    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with zipfile.ZipFile(archive_path) as archive:
        archive_members = _image_members(archive)
        member_index = {
            (_split_for_member(member), Path(PurePosixPath(member).name).stem): member
            for member in archive_members
        }
        expected = {}
        labeled_stems = {}
        for _, (split, manifest_name) in SPLITS.items():
            expected[split], labeled_stems[split] = _rewrite_manifest(
                archive,
                manifest_name,
                output_root / manifest_name,
            )
        members = []
        missing = []
        for split, stems in labeled_stems.items():
            for stem in stems:
                member = member_index.get((split, stem))
                if member is None:
                    missing.append(f"{split}/{stem}")
                else:
                    members.append(member)
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(
            f"{len(missing)} labeled images are missing from the archive: {preview}"
        )
    if len(set(members)) != len(members):
        raise ValueError("A manifest contains duplicate image names")

    actual = {split: len(stems) for split, stems in labeled_stems.items()}
    for split, row_count in expected.items():
        if actual.get(split) != row_count:
            raise ValueError(
                f"{split}: {actual[split]} images but {row_count} manifest rows"
            )
    ignored_unlabeled = len(archive_members) - len(members)

    print(
        f"Preprocessing {len(members)} KHATT images with {args.workers} workers "
        f"(target width {args.target_width}px, orientation disabled)"
    )
    records: list[dict] = []
    failures: list[dict] = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_initialize_worker,
        initargs=(
            str(archive_path),
            str(output_root),
            args.target_width,
            args.force,
        ),
    ) as pool:
        futures = {pool.submit(_process_member, member): member for member in members}
        for completed, future in enumerate(as_completed(futures), start=1):
            member = futures[future]
            try:
                records.append(future.result())
            except Exception as exc:  # keep the batch running and report every failure
                failures.append({"source": member, "error": str(exc)})
            if completed % 100 == 0 or completed == len(futures):
                elapsed = time.perf_counter() - started
                rate = completed / max(elapsed, 0.001)
                print(
                    f"[{completed}/{len(futures)}] {rate:.1f} images/s; "
                    f"failures={len(failures)}",
                    flush=True,
                )

    if failures:
        print(f"Finished with {len(failures)} failures:")
        for failure in failures[:20]:
            print(f"- {failure['source']}: {failure['error']}")
        sys.exit(1)

    processed = sum(record["status"] == "processed" for record in records)
    skipped = sum(record["status"] == "skipped" for record in records)
    elapsed = time.perf_counter() - started
    print(
        f"Completed: processed={processed}, skipped={skipped}, "
        f"unlabeled_ignored={ignored_unlabeled}, elapsed={elapsed:.2f}s"
    )

    if not args.no_archive:
        bundle_path = output_root.with_suffix(".zip")
        print(f"Creating {bundle_path.name} ...", flush=True)
        _create_archive(output_root, bundle_path)
        print(f"Created {bundle_path}")

    print(f"Fine-tuning dataset ready at {output_root}")


if __name__ == "__main__":
    main()
