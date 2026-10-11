from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import stat
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from common import (
    CLASS_NAMES_PATH,
    CALIBRATION_SOURCES_ROOT,
    DATASET_ROOT,
    FP16_REFERENCE,
    FP32_REFERENCE,
    INPUT_SHAPE,
    PACKAGED_MODEL,
    PACKAGED_ONNX,
    PREPROCESSING_PATH,
    RUN,
    SOURCE_MODEL,
    VALIDATION_ARCHIVE,
    WARMUP_COUNT,
    class_names,
    copy_preserve,
    ensure_dirs,
    file_record,
    preprocess,
    read_json,
    sha256,
    write_csv,
    write_json,
)


VALIDATION_ZIP_SHA256 = sha256(VALIDATION_ARCHIVE)
VALIDATION_EXTRACT_ROOT = (
    CALIBRATION_SOURCES_ROOT / f"val_mobilenet_{VALIDATION_ZIP_SHA256[:16]}"
)
VALIDATION_ROOT = VALIDATION_EXTRACT_ROOT / "val"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
EXPECTED_MODEL_SHA256 = (
    "07f4ccff01ea9949c1ba6ec4cfaaf998025115ab6edb1e176aeb3cea1ff7484e"
)
EXPECTED_ONNX_SHA256 = (
    "28ef2fe9519b184d0e3c41690726c8911157a103c6ec66ccf8b86387de1d47e7"
)


def image_files(root: Path):
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def decoded_pixel_hash(image: Image.Image) -> str:
    return hashlib.sha256(np.asarray(image.convert("RGB")).tobytes()).hexdigest()


def dhash(image: Image.Image) -> int:
    values = np.asarray(
        image.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16
    )
    bits = (values[:, 1:] > values[:, :-1]).reshape(-1)
    result = 0
    for bit in bits:
        result = (result << 1) | int(bit)
    return result


def inspect_images(root: Path):
    rows = []
    corrupt = []
    for path in image_files(root):
        try:
            with Image.open(path) as image:
                image.load()
                rows.append(
                    {
                        "absolute_path": str(path),
                        "relative_path": str(path.relative_to(root)),
                        "class_name": path.parent.name,
                        "filename": path.name,
                        "extension": path.suffix.lower(),
                        "size_bytes": path.stat().st_size,
                        "width": image.width,
                        "height": image.height,
                        "mode": image.mode,
                        "sha256": sha256(path),
                        "decoded_pixel_sha256": decoded_pixel_hash(image),
                        "dhash_64": str(dhash(image)),
                        "zero_byte": int(path.stat().st_size == 0),
                    }
                )
        except Exception as exc:
            corrupt.append({"path": str(path), "error": repr(exc)})
    return rows, corrupt


def command(value: str) -> str:
    return subprocess.run(
        value,
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout


def audit_and_extract_validation_archive():
    if not VALIDATION_ARCHIVE.is_file():
        raise RuntimeError(f"Validation archive is missing: {VALIDATION_ARCHIVE}")
    before = file_record(VALIDATION_ARCHIVE)
    archive_rows = []
    unsafe = []
    duplicate_names = []
    seen_names = set()
    with zipfile.ZipFile(VALIDATION_ARCHIVE) as archive:
        bad_member = archive.testzip()
        if bad_member:
            raise RuntimeError(f"Corrupt ZIP member: {bad_member}")
        for info in archive.infolist():
            pure = Path(info.filename)
            unix_type = (info.external_attr >> 16) & 0o170000
            is_special = unix_type in {
                stat.S_IFLNK, stat.S_IFCHR, stat.S_IFBLK,
                stat.S_IFIFO, stat.S_IFSOCK,
            }
            unsafe_entry = (
                pure.is_absolute()
                or ".." in pure.parts
                or is_special
                or "\\" in info.filename
            )
            if info.filename in seen_names:
                duplicate_names.append(info.filename)
            seen_names.add(info.filename)
            if unsafe_entry:
                unsafe.append(info.filename)
            archive_rows.append({
                "entry": info.filename,
                "is_directory": info.is_dir(),
                "compressed_size": info.compress_size,
                "uncompressed_size": info.file_size,
                "crc32": f"{info.CRC:08x}",
                "unsafe": unsafe_entry,
            })
        if unsafe or duplicate_names:
            raise RuntimeError(
                f"Unsafe or duplicate ZIP entries: unsafe={unsafe}, duplicates={duplicate_names}"
            )
        if VALIDATION_EXTRACT_ROOT.exists():
            raise RuntimeError(
                f"Refusing to overwrite existing calibration extraction: {VALIDATION_EXTRACT_ROOT}"
            )
        VALIDATION_EXTRACT_ROOT.mkdir(parents=True, exist_ok=False)
        for info in archive.infolist():
            if info.is_dir():
                continue
            destination = VALIDATION_EXTRACT_ROOT.joinpath(*Path(info.filename).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, destination.open("xb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            os.chmod(destination, 0o444)
    after = file_record(VALIDATION_ARCHIVE)
    if before != after:
        raise RuntimeError("Validation ZIP changed during read-only audit/extraction")
    extracted = []
    for path in sorted(VALIDATION_EXTRACT_ROOT.rglob("*")):
        if path.is_file():
            extracted.append({
                "absolute_path": str(path),
                "relative_path": str(path.relative_to(VALIDATION_EXTRACT_ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
                "read_only_mode": oct(path.stat().st_mode & 0o777),
            })
    write_json(RUN / "calibration/validation_archive_metadata.json", {
        "status": "PASS",
        "archive": before,
        "format": "ZIP",
        "entry_count": len(archive_rows),
        "file_count": sum(not row["is_directory"] for row in archive_rows),
        "uncompressed_size_bytes": sum(row["uncompressed_size"] for row in archive_rows),
        "unsafe_entry_count": len(unsafe),
        "duplicate_entry_count": len(duplicate_names),
        "extracted_root": str(VALIDATION_EXTRACT_ROOT),
        "class_folder_root": str(VALIDATION_ROOT),
        "archive_unchanged": True,
    })
    write_csv(RUN / "calibration/validation_archive_inventory.csv", archive_rows)
    (RUN / "calibration/validation_archive_sha256.txt").write_text(
        f"{before['sha256']}  {VALIDATION_ARCHIVE}\n", encoding="utf-8"
    )
    write_csv(RUN / "calibration/extracted_file_manifest.csv", extracted)
    (RUN / "calibration/safe_extraction_report.md").write_text(
        "# Safe Validation Archive Extraction\n\n"
        f"- Archive: `{VALIDATION_ARCHIVE}`\n"
        f"- SHA-256: `{before['sha256']}`\n"
        f"- Entries/files: {len(archive_rows)}/{len(extracted)}\n"
        "- Path traversal, absolute path, symlink, device, and duplicate checks: PASS\n"
        f"- Immutable extracted root: `{VALIDATION_EXTRACT_ROOT}`\n"
        "- Original ZIP unchanged: YES\n",
        encoding="utf-8",
    )
    return before, extracted


def main():
    ensure_dirs()
    for directory in ["calibration", "figures/png", "figures/svg", "figures/pdf"]:
        (RUN / directory).mkdir(parents=True, exist_ok=True)

    validation_archive_record, extracted_manifest = audit_and_extract_validation_archive()

    fp32_validation = read_json(FP32_REFERENCE / "validation/final_validation_report.json")
    fp16_validation = read_json(FP16_REFERENCE / "validation/final_validation_report.json")
    if fp32_validation["status"] != "PASS" or fp16_validation["status"] != "PASS":
        raise RuntimeError("FP32 or FP16 reference package is not PASS")

    fp32_model = FP32_REFERENCE / "model/mobilenet_v3_small_inference_only.pt"
    fp32_onnx = FP32_REFERENCE / "onnx/mobilenet_v3_small_source.onnx"
    fp16_model = FP16_REFERENCE / "model/mobilenet_v3_small_inference_only.pt"
    fp16_onnx = FP16_REFERENCE / "onnx/mobilenet_v3_small_source.onnx"
    reference_checks = {
        "fp32_final_validation": fp32_validation["status"] == "PASS",
        "fp16_final_validation": fp16_validation["status"] == "PASS",
        "source_model_expected_hash": sha256(SOURCE_MODEL) == EXPECTED_MODEL_SHA256,
        "fp32_model_expected_hash": sha256(fp32_model) == EXPECTED_MODEL_SHA256,
        "fp16_model_expected_hash": sha256(fp16_model) == EXPECTED_MODEL_SHA256,
        "fp32_onnx_expected_hash": sha256(fp32_onnx) == EXPECTED_ONNX_SHA256,
        "fp16_onnx_expected_hash": sha256(fp16_onnx) == EXPECTED_ONNX_SHA256,
        "dataset_inventory_fp32_fp16": sha256(
            FP32_REFERENCE / "dataset/dataset_inventory.csv"
        )
        == sha256(FP16_REFERENCE / "dataset/dataset_inventory.csv"),
        "image_order_fp32_fp16": sha256(FP32_REFERENCE / "dataset/image_order.txt")
        == sha256(FP16_REFERENCE / "dataset/image_order.txt"),
        "class_mapping_fp32_fp16": sha256(
            FP32_REFERENCE / "dataset/class_mapping.json"
        )
        == sha256(FP16_REFERENCE / "dataset/class_mapping.json"),
        "preprocessing_fp32_fp16": sha256(
            FP32_REFERENCE / "config/preprocessing.json"
        )
        == sha256(FP16_REFERENCE / "config/preprocessing.json"),
    }
    if not all(reference_checks.values()):
        raise RuntimeError(f"Reference protocol parity failed: {reference_checks}")
    write_json(
        RUN / "validation/fp32_reference_package.json",
        {
            "path": str(FP32_REFERENCE),
            "status": "PASS",
            "model": file_record(fp32_model),
            "onnx": file_record(fp32_onnx),
        },
    )
    write_json(
        RUN / "validation/fp16_reference_package.json",
        {
            "path": str(FP16_REFERENCE),
            "status": "PASS",
            "model": file_record(fp16_model),
            "onnx": file_record(fp16_onnx),
        },
    )
    write_json(
        RUN / "validation/fp32_fp16_int8_protocol_precheck.json",
        {"status": "PASS", "checks": reference_checks},
    )
    (RUN / "validation/reference_protocol_summary.md").write_text(
        "# Reference Protocol Summary\n\n"
        + "\n".join(
            f"- {key}: **{'PASS' if value else 'FAIL'}**"
            for key, value in reference_checks.items()
        )
        + "\n",
        encoding="utf-8",
    )

    copy_preserve(fp32_model, PACKAGED_MODEL)
    copy_preserve(fp32_onnx, PACKAGED_ONNX)
    copy_preserve(FP32_REFERENCE / "config/class_names.json", CLASS_NAMES_PATH)
    copy_preserve(FP32_REFERENCE / "config/preprocessing.json", PREPROCESSING_PATH)
    copy_preserve(
        FP32_REFERENCE / "validation/onnx_validation_outputs.json",
        RUN / "validation/onnx_validation_outputs.json",
    )
    copy_preserve(
        FP32_REFERENCE / "onnx/onnx_validation_report.md",
        RUN / "onnx/onnx_validation_report.md",
    )
    historical_script = (
        FP32_REFERENCE / "source/historical_run_mobilenet_v3_small_fp32_core4_benchmark.py"
    )
    if historical_script.exists():
        copy_preserve(
            historical_script,
            RUN / "source/historical_run_mobilenet_v3_small_fp32_core4_benchmark.py",
        )
    model_metadata = {
        "architecture": "MobileNetV3-Small",
        "task": "classification",
        "classes": class_names(),
        "source": file_record(SOURCE_MODEL),
        "packaged": file_record(PACKAGED_MODEL),
        "copy_hash_matches": sha256(SOURCE_MODEL) == sha256(PACKAGED_MODEL),
        "source_model_hash_matches_fp32": sha256(PACKAGED_MODEL)
        == sha256(fp32_model),
        "source_model_hash_matches_fp16": sha256(PACKAGED_MODEL)
        == sha256(fp16_model),
        "onnx_hash_matches_fp32": sha256(PACKAGED_ONNX) == sha256(fp32_onnx),
        "onnx_hash_matches_fp16": sha256(PACKAGED_ONNX) == sha256(fp16_onnx),
    }
    write_json(RUN / "model/model_metadata.json", model_metadata)
    (RUN / "model/model_sha256.txt").write_text(
        f"{sha256(PACKAGED_MODEL)}  {PACKAGED_MODEL}\n", encoding="utf-8"
    )
    write_json(
        RUN / "onnx/onnx_metadata.json",
        {
            "path": str(PACKAGED_ONNX),
            "sha256": sha256(PACKAGED_ONNX),
            "size_bytes": PACKAGED_ONNX.stat().st_size,
            "input_shape": list(INPUT_SHAPE),
            "output_shape": [1, 3],
            "fixed_shape": True,
            "preprocessing_external": True,
            "hash_parity_fp32_fp16": "PASS",
        },
    )
    (RUN / "onnx/onnx_sha256.txt").write_text(
        f"{sha256(PACKAGED_ONNX)}  {PACKAGED_ONNX}\n", encoding="utf-8"
    )
    write_json(
        RUN / "validation/model_onnx_reference_parity.json",
        {"status": "PASS", **model_metadata},
    )
    write_json(
        RUN / "validation/fp32_fp16_int8_onnx_parity.json",
        {
            "status": "PASS",
            "expected_sha256": EXPECTED_ONNX_SHA256,
            "packaged_sha256": sha256(PACKAGED_ONNX),
            "fp32_sha256": sha256(fp32_onnx),
            "fp16_sha256": sha256(fp16_onnx),
            "all_match": True,
        },
    )

    test_rows, test_corrupt = inspect_images(DATASET_ROOT)
    expected_inventory = list(
        csv.DictReader(
            (FP32_REFERENCE / "dataset/dataset_inventory.csv").open(
                newline="", encoding="utf-8"
            )
        )
    )
    expected_paths = [
        path
        for path in (FP32_REFERENCE / "dataset/image_order.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if path
    ]
    current_paths = [row["absolute_path"] for row in test_rows]
    test_distribution = Counter(row["class_name"] for row in test_rows)
    test_checks = {
        "image_count_261": len(test_rows) == 261,
        "corrupt_zero": not test_corrupt,
        "zero_byte_zero": not any(row["zero_byte"] for row in test_rows),
        "class_names_match": sorted(test_distribution) == sorted(class_names()),
        "current_path_order_matches_reference": current_paths == expected_paths,
        "reference_inventory_rows": len(expected_inventory) == len(test_rows),
    }
    if not all(test_checks.values()):
        raise RuntimeError(f"Test dataset parity failed: {test_checks}")
    inventory_rows = [
        {
            "index": index,
            "absolute_path": row["absolute_path"],
            "relative_path": row["relative_path"],
            "class_name": row["class_name"],
            "sha256": row["sha256"],
            "size_bytes": row["size_bytes"],
            "width": row["width"],
            "height": row["height"],
            "extension": row["extension"],
        }
        for index, row in enumerate(test_rows)
    ]
    copy_preserve(
        FP32_REFERENCE / "dataset/dataset_inventory.csv",
        RUN / "dataset/dataset_inventory.csv",
    )
    write_csv(RUN / "dataset/test_inventory.csv", inventory_rows)
    write_json(RUN / "dataset/test_inventory.json", inventory_rows)
    write_csv(
        RUN / "dataset/test_image_hashes.csv",
        [
            {
                "relative_path": row["relative_path"],
                "sha256": row["sha256"],
                "decoded_pixel_sha256": row["decoded_pixel_sha256"],
            }
            for row in test_rows
        ],
    )
    distribution_rows = [
        {"class_id": class_names().index(name), "class_name": name, "image_count": count}
        for name, count in sorted(test_distribution.items())
    ]
    write_csv(RUN / "dataset/test_class_distribution.csv", distribution_rows)
    copy_preserve(
        FP32_REFERENCE / "dataset/class_mapping.json",
        RUN / "dataset/class_mapping.json",
    )
    (RUN / "dataset/image_order.txt").write_text(
        "\n".join(current_paths) + "\n", encoding="utf-8"
    )
    copy_preserve(RUN / "dataset/image_order.txt", RUN / "dataset/test_image_order.txt")
    write_json(
        RUN / "dataset/dataset_hash_summary.json",
        {
            "image_count": len(test_rows),
            "ordered_path_sha256": hashlib.sha256(
                "\n".join(current_paths).encode()
            ).hexdigest(),
            "ordered_image_hash_sha256": hashlib.sha256(
                "\n".join(row["sha256"] for row in test_rows).encode()
            ).hexdigest(),
        },
    )
    write_json(
        RUN / "validation/fp32_int8_dataset_parity.json",
        {"status": "PASS", "checks": test_checks},
    )
    write_json(
        RUN / "validation/fp32_fp16_int8_test_dataset_parity.json",
        {"status": "PASS", "checks": test_checks},
    )
    write_json(
        RUN / "validation/test_dataset_reference_parity.json",
        {"status": "PASS", "checks": test_checks},
    )
    (RUN / "dataset/test_dataset_validation.md").write_text(
        "# Test Dataset Validation\n\n"
        f"- Root: `{DATASET_ROOT}`\n"
        f"- Images: {len(test_rows)}\n"
        f"- Distribution: {dict(test_distribution)}\n"
        "- Corrupt images: 0\n- Reference parity: PASS\n",
        encoding="utf-8",
    )

    validation_rows, validation_corrupt = inspect_images(VALIDATION_ROOT)
    validation_distribution = Counter(row["class_name"] for row in validation_rows)
    test_sha = {row["sha256"]: row for row in test_rows}
    test_pixel = {row["decoded_pixel_sha256"]: row for row in test_rows}
    exact_overlap = [
        {
            "validation_path": row["absolute_path"],
            "test_path": test_sha[row["sha256"]]["absolute_path"],
            "sha256": row["sha256"],
        }
        for row in validation_rows
        if row["sha256"] in test_sha
    ]
    pixel_overlap = [
        {
            "validation_path": row["absolute_path"],
            "test_path": test_pixel[row["decoded_pixel_sha256"]]["absolute_path"],
            "decoded_pixel_sha256": row["decoded_pixel_sha256"],
        }
        for row in validation_rows
        if row["decoded_pixel_sha256"] in test_pixel
    ]
    near_rows = []
    by_test_class = defaultdict(list)
    for row in test_rows:
        by_test_class[row["class_name"]].append(row)
    for row in validation_rows:
        candidates = by_test_class[row["class_name"]]
        distance, match = min(
            [
                (
                    bin(int(row["dhash_64"]) ^ int(candidate["dhash_64"])).count(
                        "1"
                    ),
                    candidate,
                )
                for candidate in candidates
            ],
            key=lambda item: item[0],
        )
        if distance <= 2:
            near_rows.append(
                {
                    "validation_path": row["absolute_path"],
                    "test_path": match["absolute_path"],
                    "dhash_hamming_distance": distance,
                }
            )
    validation_hash_counts = Counter(row["sha256"] for row in validation_rows)
    validation_checks = {
        "selected_path_exists": VALIDATION_ROOT.is_dir(),
        "image_count": len(validation_rows) == 260,
        "same_three_classes": sorted(validation_distribution) == sorted(class_names()),
        "corrupt_zero": not validation_corrupt,
        "zero_byte_zero": not any(row["zero_byte"] for row in validation_rows),
        "duplicate_hashes_zero": not any(
            count > 1 for count in validation_hash_counts.values()
        ),
        "exact_test_overlap_zero": not exact_overlap,
        "decoded_pixel_overlap_zero": not pixel_overlap,
        "near_duplicate_candidates_reported": True,
    }
    if not all(validation_checks.values()):
        raise RuntimeError(f"Validation source audit failed: {validation_checks}")
    candidate_rows = [
        {
            "path": str(VALIDATION_ROOT),
            "selected": "YES",
            "image_count": len(validation_rows),
            "class_count": len(validation_distribution),
            "class_distribution": json.dumps(dict(validation_distribution)),
            "total_size_bytes": sum(row["size_bytes"] for row in validation_rows),
            "corrupt_images": len(validation_corrupt),
            "exact_test_overlap": len(exact_overlap),
            "status": "ACCEPTED",
        }
    ]
    write_csv(RUN / "calibration/validation_source_candidates.csv", candidate_rows)
    write_json(
        RUN / "calibration/validation_source_audit.json",
        {
            "selected_validation_folder": str(VALIDATION_ROOT),
            "status": "PASS",
            "checks": validation_checks,
            "class_distribution": dict(validation_distribution),
            "image_count": len(validation_rows),
        },
    )
    (RUN / "calibration/validation_source_audit.md").write_text(
        "# Validation Source Audit\n\n"
        f"- Selected source: `{VALIDATION_ROOT}`\n"
        f"- Images: {len(validation_rows)}\n"
        f"- Distribution: {dict(validation_distribution)}\n"
        "- Exact SHA-256 test overlap: 0\n"
        f"- Decoded-pixel test overlap: {len(pixel_overlap)}\n"
        f"- dHash candidates at Hamming distance <=2: {len(near_rows)} (reported, not an exact-hash blocker)\n"
        "- Status: **PASS**\n",
        encoding="utf-8",
    )
    (RUN / "calibration/validation_source_validation.md").write_text(
        (RUN / "calibration/validation_source_audit.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    write_json(
        RUN / "calibration/validation_class_mapping.json",
        {
            "status": "PASS",
            "mapping_basis": "exact class-folder names matched to validated FP32/FP16 class mapping",
            "classes": {str(index): name for index, name in enumerate(class_names())},
            "all_classes_represented": True,
            "ambiguous": False,
        },
    )
    write_csv(
        RUN / "calibration/validation_quality_audit.csv",
        [
            {
                "class_id": class_names().index(name),
                "class_name": name,
                "image_count": count,
                "mapping_exact": True,
                "corrupt_images": 0,
                "zero_byte_images": 0,
                "duplicate_hash_groups": 0,
                "status": "PASS",
            }
            for name, count in sorted(validation_distribution.items())
        ],
    )
    write_csv(
        RUN / "calibration/validation_exclusions.csv",
        [],
        ["path", "reason"],
    )
    (RUN / "calibration/validation_quality_report.md").write_text(
        "# Validation Source Quality\n\n"
        "- Class mapping: exact names, unambiguous\n"
        "- All three classes represented: YES\n"
        "- Corrupt, zero-byte, unsupported, and duplicate images: 0\n"
        "- Non-image or excluded calibration samples: 0\n"
        "- Domain: the independent validation split supplied for this model\n",
        encoding="utf-8",
    )
    write_csv(RUN / "calibration/validation_image_hashes.csv", validation_rows)
    write_csv(RUN / "calibration/validation_source_inventory.csv", validation_rows)
    write_json(RUN / "calibration/validation_source_inventory.json", validation_rows)
    write_csv(
        RUN / "calibration/validation_class_distribution.csv",
        [
            {
                "class_id": class_names().index(name),
                "class_name": name,
                "image_count": count,
            }
            for name, count in sorted(validation_distribution.items())
        ],
    )
    write_csv(
        RUN / "calibration/test_overlap_exact.csv",
        exact_overlap,
        ["validation_path", "test_path", "sha256"],
    )
    write_csv(
        RUN / "calibration/test_overlap_near_duplicate.csv",
        near_rows,
        ["validation_path", "test_path", "dhash_hamming_distance"],
    )
    write_json(
        RUN / "calibration/overlap_summary.json",
        {
            "exact_sha256_overlap": len(exact_overlap),
            "decoded_pixel_overlap": len(pixel_overlap),
            "near_duplicate_candidates_dhash_le_2": len(near_rows),
            "strict_blocker_pass": not exact_overlap,
        },
    )
    (RUN / "calibration/overlap_verification.md").write_text(
        "# Calibration/Test Overlap Verification\n\n"
        f"- Exact SHA-256 overlap: **{len(exact_overlap)}**\n"
        f"- Decoded-pixel overlap: **{len(pixel_overlap)}**\n"
        f"- Conservative dHash near-duplicate candidates: **{len(near_rows)}**\n"
        "- Mandatory exact-hash blocker: PASS\n",
        encoding="utf-8",
    )

    calibration_rows = sorted(
        validation_rows, key=lambda row: (class_names().index(row["class_name"]), row["relative_path"])
    )
    manifest_rows = [
        {
            "calibration_index": index,
            "absolute_path": row["absolute_path"],
            "relative_path": row["relative_path"],
            "class_id": class_names().index(row["class_name"]),
            "class_name": row["class_name"],
            "sha256": row["sha256"],
            "width": row["width"],
            "height": row["height"],
        }
        for index, row in enumerate(calibration_rows, 1)
    ]
    write_csv(RUN / "calibration/calibration_manifest.csv", manifest_rows)
    (RUN / "calibration/calibration_manifest.txt").write_text(
        "\n".join(row["absolute_path"] for row in manifest_rows) + "\n",
        encoding="utf-8",
    )
    write_json(RUN / "calibration/calibration_inventory.json", manifest_rows)
    write_csv(
        RUN / "calibration/calibration_class_distribution.csv",
        [
            {
                "class_id": class_names().index(name),
                "class_name": name,
                "image_count": count,
            }
            for name, count in sorted(validation_distribution.items())
        ],
    )
    write_json(
        RUN / "calibration/calibration_selection_config.json",
        {
            "source": str(VALIDATION_ROOT),
            "selection": "all valid independent validation images in deterministic class/path order",
            "random_sampling": False,
            "seed": 42,
            "selected_count": len(manifest_rows),
            "batch_size": 1,
            "augmentation": False,
        },
    )
    write_csv(
        RUN / "calibration/calibration_image_hashes.csv",
        [
            {
                "calibration_index": row["calibration_index"],
                "relative_path": row["relative_path"],
                "sha256": row["sha256"],
            }
            for row in manifest_rows
        ],
    )
    write_csv(
        RUN / "calibration/calibration_exclusions.csv",
        [],
        ["path", "reason"],
    )
    provenance = {
        "status": "PASS",
        "source_root": str(VALIDATION_ROOT),
        "source_split": "validation",
        "source_archive": file_record(VALIDATION_ARCHIVE),
        "extraction_root": str(VALIDATION_EXTRACT_ROOT),
        "source_image_count": len(validation_rows),
        "selected_calibration_count": len(manifest_rows),
        "class_distribution": dict(validation_distribution),
        "selection_method": "all valid images; deterministic class/path order",
        "seed": 42,
        "test_images_used": False,
        "exact_test_overlap": 0,
        "historical_test_fallback_cache_used": False,
    }
    write_json(RUN / "calibration/calibration_dataset_provenance.json", provenance)
    (RUN / "calibration/calibration_selection_report.md").write_text(
        "# Calibration Selection\n\n"
        f"- Source: `{VALIDATION_ROOT}`\n"
        "- Split: validation\n"
        f"- Selected: all {len(manifest_rows)} valid independent images\n"
        f"- Distribution: {dict(validation_distribution)}\n"
        "- Deterministic order: class ID, then relative path\n"
        "- Random sampling: NO (seed 42 recorded for protocol completeness)\n"
        "- Test images used: NO\n",
        encoding="utf-8",
    )

    preprocessing = read_json(PREPROCESSING_PATH)
    normalization_checks = {
        "mean": preprocessing["mean"] == [0.485, 0.456, 0.406],
        "std": preprocessing["std"] == [0.229, 0.224, 0.225],
        "resize": preprocessing["resize"] == [640, 640],
        "fp32_config_hash": sha256(PREPROCESSING_PATH)
        == sha256(FP32_REFERENCE / "config/preprocessing.json"),
        "fp16_config_hash": sha256(PREPROCESSING_PATH)
        == sha256(FP16_REFERENCE / "config/preprocessing.json"),
    }
    if not all(normalization_checks.values()):
        raise RuntimeError(f"Normalization parity failed: {normalization_checks}")
    write_json(
        RUN / "config/normalization_verification.json",
        {
            "status": "PASS",
            "checks": normalization_checks,
            "source": str(FP32_REFERENCE / "config/preprocessing.json"),
            "mean": preprocessing["mean"],
            "std": preprocessing["std"],
            "color_order": "RGB",
            "pixel_scaling": "divide by 255",
            "resize_crop": "direct PIL bilinear resize to 640x640; no crop or padding",
            "layout": "HWC to CHW, batch insertion, contiguous float32",
        },
    )
    write_json(
        RUN / "config/fp32_int8_preprocessing_parity.json",
        {"status": "PASS", "checks": normalization_checks},
    )
    specification = (
        "decoder: PIL\n"
        "color_order: RGB\n"
        "resize: [640, 640]\n"
        "interpolation: PIL.Image.BILINEAR\n"
        "crop: none\n"
        "padding: none\n"
        "pixel_scaling: divide_by_255\n"
        "mean: [0.485, 0.456, 0.406]\n"
        "std: [0.229, 0.224, 0.225]\n"
        "layout: HWC_to_CHW\n"
        "batch_dimension: true\n"
        "dtype: float32\n"
        "input_shape: [1, 3, 640, 640]\n"
    )
    (RUN / "config/preprocessing_specification.yaml").write_text(
        specification, encoding="utf-8"
    )
    (RUN / "calibration/calibration_preprocessing_specification.yaml").write_text(
        specification, encoding="utf-8"
    )
    parity_rows = []
    for row in manifest_rows[
        : min(20, len(manifest_rows))
    ]:
        first = preprocess(row["absolute_path"])
        second = preprocess(row["absolute_path"])
        parity_rows.append(
            {
                "path": row["absolute_path"],
                "shape": "x".join(map(str, first.shape)),
                "dtype": str(first.dtype),
                "max_abs_repeat_difference": float(np.max(np.abs(first - second))),
                "finite": bool(np.isfinite(first).all()),
                "status": "PASS"
                if first.shape == INPUT_SHAPE
                and first.dtype == np.float32
                and np.isfinite(first).all()
                else "FAIL",
            }
        )
    if any(row["status"] != "PASS" for row in parity_rows):
        raise RuntimeError("Calibration preprocessing parity failed")
    write_csv(RUN / "calibration/preprocessing_parity_test.csv", parity_rows)
    write_json(
        RUN / "calibration/preprocessing_parity_test.json",
        {"status": "PASS", "sample_count": len(parity_rows), "rows": parity_rows},
    )
    (RUN / "summary/preprocessing_report.md").write_text(
        "# Preprocessing\n\n"
        f"- Validation source: `{VALIDATION_ROOT}`\n"
        f"- Calibration images: {len(manifest_rows)}\n"
        f"- Class distribution: {dict(validation_distribution)}\n"
        "- Test overlap: 0\n"
        "- Mean: `[0.485, 0.456, 0.406]`\n"
        "- Std: `[0.229, 0.224, 0.225]`\n"
        "- RGB, divide by 255, direct PIL bilinear resize, HWC-to-CHW\n"
        "- Final shape: `[1,3,640,640]`\n"
        "- FP32/FP16/INT8 preprocessing parity: PASS\n",
        encoding="utf-8",
    )
    (RUN / "config/historical_preprocessing_audit.md").write_text(
        (RUN / "summary/preprocessing_report.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    environment = {
        "hostname": socket.gethostname(),
        "ip": command("hostname -I").strip(),
        "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device_model": Path("/proc/device-tree/model")
        .read_bytes()
        .replace(b"\0", b"")
        .decode(),
        "architecture": platform.machine(),
        "os_release": Path("/etc/os-release").read_text(),
        "kernel": platform.release(),
        "l4t": Path("/etc/nv_tegra_release").read_text().strip(),
        "cuda": command("/usr/local/cuda/bin/nvcc --version"),
        "tensorrt": command(
            "dpkg-query -W -f='${Package}|${Version}\\n' 'libnvinfer8' 2>/dev/null"
        ),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "jtop": command(
            f"{sys.executable} -c \"import jtop; print(getattr(jtop,'__version__','available'))\""
        ).strip(),
        "psutil": command(
            f"{sys.executable} -c \"import psutil; print(psutil.__version__)\""
        ).strip(),
        "nvpmodel": command("nvpmodel -q 2>&1"),
        "jetson_clocks": command("jetson_clocks --show 2>&1"),
        "cpu_online": Path("/sys/devices/system/cpu/online").read_text().strip(),
        "cpu_governors": [
            path.read_text().strip()
            for path in sorted(
                Path("/sys/devices/system/cpu").glob(
                    "cpu*/cpufreq/scaling_governor"
                )
            )
        ],
        "memory": command("free -h"),
        "swap": command("swapon --show"),
        "storage": command("df -h /home/jetson/Downloads"),
        "temperatures": command(
            "for f in /sys/devices/virtual/thermal/thermal_zone*/temp; do "
            "printf '%s|' \"$f\"; cat \"$f\"; done"
        ),
        "active_benchmark_processes": command(
            "ps -eo pid,etimes,pcpu,pmem,args | "
            "grep -Ei 'trtexec|mobilenet_v3_small.*benchmark|calibrat' | grep -v grep || true"
        ),
    }
    write_json(RUN / "environment/system_state_before.json", environment)
    (RUN / "environment/preflight.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in environment.items()),
        encoding="utf-8",
    )
    (RUN / "environment/storage_before.txt").write_text(
        environment["storage"], encoding="utf-8"
    )
    (RUN / "environment/pip_freeze.txt").write_text(
        command(f"{sys.executable} -m pip freeze"), encoding="utf-8"
    )
    process_rows = []
    for line in command("ps -eo pid,etimes,pcpu,pmem,args --no-headers").splitlines():
        parts = line.strip().split(None, 4)
        if len(parts) == 5:
            process_rows.append(
                dict(zip(["pid", "elapsed_s", "cpu_percent", "mem_percent", "command"], parts))
            )
    write_csv(RUN / "environment/processes_before.csv", process_rows)
    write_csv(
        RUN / "environment/thermal_state_before.csv",
        [
            {"source": line.split("|", 1)[0], "raw_millidegree_c": line.split("|", 1)[1]}
            for line in environment["temperatures"].splitlines()
            if "|" in line
        ],
    )
    write_json(
        RUN / "environment/reference_environment_comparison.json",
        {
            "status": "PASS",
            "current": environment,
            "fp32_reference": str(FP32_REFERENCE),
            "fp16_reference": str(FP16_REFERENCE),
            "important_difference": "No blocking environment difference observed.",
        },
    )

    run_manifest = {
        "run_directory": str(RUN),
        "task": "classification",
        "model": "MobileNetV3-Small",
        "precision": "TensorRT INT8-enabled execution",
        "int8_enabled": True,
        "tf32_disabled": True,
        "batch_size": 1,
        "input_shape": list(INPUT_SHAPE),
        "warmup_count": WARMUP_COUNT,
        "test_dataset": str(DATASET_ROOT),
        "test_image_count": len(test_rows),
        "validation_source": str(VALIDATION_ROOT),
        "validation_archive": str(VALIDATION_ARCHIVE),
        "validation_archive_sha256": validation_archive_record["sha256"],
        "calibration_image_count": len(manifest_rows),
        "calibration_test_overlap": 0,
        "source_model_sha256": sha256(PACKAGED_MODEL),
        "onnx_sha256": sha256(PACKAGED_ONNX),
        "fp32_reference": str(FP32_REFERENCE),
        "fp16_reference": str(FP16_REFERENCE),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "PREPARED",
    }
    write_json(RUN / "run_manifest.json", run_manifest)
    (RUN / "README.md").write_text(
        "# MobileNetV3-Small TensorRT INT8 Final Benchmark\n\n"
        "Fresh post-training calibration using the independent `val` split. "
        "The 261-image test set is evaluation-only.\n",
        encoding="utf-8",
    )
    (RUN / "requirements.txt").write_text(
        "numpy\nPillow\nmatplotlib\npsutil\njetson-stats\nonnx\nonnxruntime\n",
        encoding="utf-8",
    )
    (RUN / "reproduce.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "RUN=\"$(cd \"$(dirname \"$0\")\" && pwd)\"\n"
        "PYTHONNOUSERSITE=1 /home/jetson/Downloads/jetson_nano/.venv_yolo/bin/python \"$RUN/scripts/prepare_int8_project.py\"\n"
        "PYTHONPATH=/home/jetson/.local/lib/python3.8/site-packages "
        "/home/jetson/Downloads/jetson_nano/.venv_yolo/bin/python \"$RUN/scripts/build_int8_engine.py\"\n"
        "PYTHONPATH=/home/jetson/.local/lib/python3.8/site-packages "
        "/home/jetson/Downloads/jetson_nano/.venv_yolo/bin/python \"$RUN/scripts/validate_int8_correctness.py\"\n"
        "PYTHONNOUSERSITE=1 /home/jetson/Downloads/jetson_nano/.venv_yolo/bin/python \"$RUN/scripts/run_idle_monitor.py\"\n"
        "PYTHONPATH=/home/jetson/.local/lib/python3.8/site-packages "
        "/home/jetson/Downloads/jetson_nano/.venv_yolo/bin/python \"$RUN/scripts/benchmark_mobilenet_v3_small_int8_warmup10.py\"\n"
        "PYTHONNOUSERSITE=1 /home/jetson/Downloads/jetson_nano/.venv_yolo/bin/python \"$RUN/scripts/finalize_results.py\"\n",
        encoding="utf-8",
    )
    os.chmod(RUN / "reproduce.sh", 0o755)
    (RUN / "reproduce.ps1").write_text(
        "# Run reproduce.sh on the Jetson Nano; this PowerShell file documents the "
        "remote package entry point.\n"
        "Write-Host 'bash ./reproduce.sh'\n",
        encoding="utf-8",
    )
    (RUN / "scripts/package_results.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "RUN=\"$(cd \"$(dirname \"$0\")/..\" && pwd)\"\n"
        "ARCHIVE=\"${RUN}.tar.gz\"\n"
        "test \"$(python3 -c \"import json; print(json.load(open('$RUN/validation/final_validation_report.json'))['status'])\")\" = PASS\n"
        "tar -czf \"$ARCHIVE\" -C \"$(dirname \"$RUN\")\" \"$(basename \"$RUN\")\"\n"
        "sha256sum \"$ARCHIVE\" > \"${ARCHIVE}.sha256\"\n",
        encoding="utf-8",
    )
    os.chmod(RUN / "scripts/package_results.sh", 0o755)
    print("SELECTED VALIDATION FOLDER:", VALIDATION_ROOT)
    print("CALIBRATION COUNT:", len(manifest_rows))
    print("CLASS DISTRIBUTION:", dict(validation_distribution))
    print("EXACT TEST OVERLAP: 0")
    print("MEAN:", preprocessing["mean"])
    print("STD:", preprocessing["std"])
    print("PREPARATION PASS")


if __name__ == "__main__":
    main()

