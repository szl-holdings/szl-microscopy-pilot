"""Offline HistoForge export intake for SZL's evidence-gated microscopy training lane.

Recognizes HistoForge's JCS-digested *format* (not a digital signature):
dataset.json schema_version=2 inside an exact ZIP, plus an exported split
sidecar. Validates a small, bounded package without extraction or code/model
execution. This is NOT the upstream's full scientific dataset validator and
NEVER dispatches a trainer, downloads weights or approves data rights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
import uuid
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import rfc8785

INTAKE_SCHEMA = "szl.histoforge-intake-request/v1"
RESULT_SCHEMA = "szl.histoforge-intake-evidence/v1"
UPSTREAM_BACKEND_SHA = "ec0a838dcc875d5c01c68c6c585747bc615fd6cf"
UPSTREAM_MODELS_SHA = "7c299f75d0dc15a10318b49c85285f490fa476e0"
UPSTREAM_EXTENSION_SHA = "147a0813cd2682a8a38aa88f31c4c38f35c10b2a"
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
REPO = re.compile(r"[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+\Z")
MAX_REQUEST = 32 * 1024
MAX_JSON = 2 * 1024 * 1024
MAX_ARCHIVE = 128 * 1024 * 1024
MAX_MEMBER = 64 * 1024 * 1024
MAX_TOTAL_UNCOMPRESSED = 128 * 1024 * 1024
MAX_MEMBERS = 4000
MAX_IMAGES = 512
MAX_ROIS = 2000
MAX_GROUPS = 512
BLOCK_SIZE = 256 * 1024

REQUIRED_REQUEST = {
    "schema",
    "upstream",
    "archive",
    "split_sidecar",
    "module",
    "permissions",
}
UPSTREAM_KEYS = {"backend_commit", "models_commit", "extension_commit"}
FILE_KEYS = {"path", "size_bytes", "sha256"}
MODULE_KEYS = {"repo", "revision", "model_id", "weight_terms"}
PERMISSIONS_KEYS = {
    "local_training_allowed",
    "cloud_upload_allowed",
    "specimen_identity_reviewed",
    "publication_allowed",
}
MANIFEST_KEYS = {
    "schema_version",
    "dataset_digest",
    "task",
    "artifacts",
    "project_groups",
    "images",
    "rois",
    "files",
}
MANIFEST_FILE_KEYS = {"path", "size_bytes", "sha256"}
SPLIT_KEYS = {"schema_version", "dataset_digest", "split_plan"}
SPLIT_PLAN_KEYS = {"type", "unit", "assignment_source", "units"}
SPLIT_PLAN_GENERATED_KEYS = SPLIT_PLAN_KEYS | {"requested_validation_percent"}
ARTIFACT_REP = {
    "png": "source.png",
    "ome_bigtiff": "source.ome.tif",
    "original": "source.tif",
}


class IntakeHold(ValueError):
    """Stable refusal code, never an uploaded patient/sample identifier."""


def hold(reason: str) -> None:
    raise IntakeHold(reason)


def require_keys(value: Any, expected: set[str], reason: str) -> dict:
    if not isinstance(value, dict) or set(value) != expected:
        hold(reason)
    return value


def require_count(value: Any, minimum: int, maximum: int, reason: str) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        hold(reason)
    return value


def require_sha(value: Any, pattern: re.Pattern, reason: str) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        hold(reason)
    return value


def read_bounded(path: Path, limit: int) -> bytes:
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        hold("JSON_BYTE_BOUNDARY")
    return raw


def safe_json(raw: bytes, size_limit: int) -> Any:
    if not isinstance(raw, bytes) or not 0 < len(raw) <= size_limit:
        hold("JSON_BYTE_BOUNDARY")

    def unique(pairs: list) -> dict:
        result = {}
        for name, value in pairs:
            if name in result:
                hold("DUPLICATE_JSON_KEY")
            result[name] = value
        return result

    def reject_constant(_: str) -> None:
        hold("NONFINITE_JSON")

    try:
        return json.loads(raw, object_pairs_hook=unique, parse_constant=reject_constant)
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        if isinstance(exc, IntakeHold):
            raise
        hold("INVALID_JSON")


def jcs(value: Any) -> bytes:
    try:
        return rfc8785.dumps(value)
    except (TypeError, ValueError, OverflowError):
        hold("JCS_DOMAIN_UNSUPPORTED")


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def safe_relative(value: Any) -> PurePosixPath:
    if (
        not isinstance(value, str)
        or len(value) > 180
        or not value.isascii()
        or "\\" in value
        or value.startswith("/")
        or "//" in value
    ):
        hold("UNSAFE_RELATIVE_PATH")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        hold("UNSAFE_RELATIVE_PATH")
    return PurePosixPath(*parts)


def local_file(root: Path, spec: Any) -> Path:
    require_keys(spec, FILE_KEYS, "INPUT_FILE_SCHEMA")
    relative = safe_relative(spec["path"])
    require_sha(spec["sha256"], HEX64, "INPUT_DIGEST_INVALID")
    require_count(spec["size_bytes"], 1, MAX_ARCHIVE, "INPUT_SIZE_INVALID")
    path = root.joinpath(*relative.parts)
    # No symlinks in the path, even when their resolved target stays under root.
    walk = root
    for part in relative.parts:
        walk = walk / part
        if walk.is_symlink():
            hold("SYMLINK_INPUT_FORBIDDEN")
    if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        hold("INPUT_FILE_UNAVAILABLE")
    if path.stat().st_size != spec["size_bytes"]:
        hold("INPUT_BYTES_DIFFER")
    return path


def file_digest(path: Path, bound: int) -> str:
    h = hashlib.sha256()
    total = 0
    with path.open("rb") as stream:
        while True:
            block = stream.read(BLOCK_SIZE)
            if not block:
                break
            total += len(block)
            if total > bound:
                hold("FILE_READ_CAP")
            h.update(block)
    return h.hexdigest()


def verify_input_file(root: Path, spec: Any) -> Path:
    file = local_file(root, spec)
    if file_digest(file, MAX_ARCHIVE) != spec["sha256"]:
        hold("INPUT_SHA256_MISMATCH")
    return file


def manifest_checks(manifest: Any) -> tuple[dict, set[str]]:
    m = require_keys(manifest, MANIFEST_KEYS, "DATASET_MANIFEST_SCHEMA")
    if m["schema_version"] != 2 or type(m["schema_version"]) is not int:
        hold("UNSUPPORTED_HISTOFORGE_DATASET_SCHEMA")
    require_sha(m["dataset_digest"], HEX64, "DATASET_DIGEST_INVALID")
    preimage = {key: value for key, value in m.items() if key != "dataset_digest"}
    if sha_bytes(jcs(preimage)) != m["dataset_digest"]:
        hold("DATASET_JCS_DIGEST_MISMATCH")
    task = require_keys(
        m["task"],
        {
            "capability_id",
            "input_kind",
            "model_output",
            "supports_negative_rois",
            "model_microns_per_pixel",
            "annotation_class_map",
        },
        "TASK_SCHEMA",
    )
    mpp = task["model_microns_per_pixel"]
    if type(mpp) not in {int, float} or not math.isfinite(mpp) or mpp <= 0 or mpp > 100:
        hold("MICRONS_PER_PIXEL_INVALID")
    if task["input_kind"] not in {"field", "crop"} or not isinstance(
        task["model_output"], dict
    ):
        hold("TASK_INPUT_OR_OUTPUT_UNSUPPORTED")
    if type(task["supports_negative_rois"]) is not bool or not isinstance(
        task["annotation_class_map"], dict
    ):
        hold("TASK_LABEL_MAPPING_UNAVAILABLE")
    groups = m["project_groups"]
    images = m["images"]
    rois = m["rois"]
    artifacts = m["artifacts"]
    files = m["files"]
    if (
        not isinstance(groups, list)
        or not 2 <= len(groups) <= MAX_GROUPS
        or any(not isinstance(g, str) or not g for g in groups)
        or len(set(groups)) != len(groups)
    ):
        hold("SPECIMEN_GROUPS_UNAVAILABLE")
    if groups != sorted(groups, key=jcs):
        hold("PROJECT_GROUP_ORDER_NOT_CANONICAL")
    if not isinstance(images, list) or not 2 <= len(images) <= MAX_IMAGES:
        hold("IMAGE_COVERAGE_UNAVAILABLE")
    if not isinstance(rois, list) or not 2 <= len(rois) <= MAX_ROIS:
        hold("ROI_COVERAGE_UNAVAILABLE")
    if not isinstance(artifacts, list) or not 2 <= len(artifacts) <= MAX_ROIS:
        hold("ARTIFACT_COVERAGE_UNAVAILABLE")
    if not isinstance(files, list) or not 1 <= len(files) <= MAX_MEMBERS - 1:
        hold("FILE_INVENTORY_UNAVAILABLE")
    image_ids = []
    image_groups = set()
    for image in images:
        require_keys(
            image,
            {"project_entry_id_digest", "project_group_index"},
            "IMAGE_GROUP_BINDING_REQUIRED",
        )
        image_ids.append(
            require_sha(
                image["project_entry_id_digest"], HEX64, "IMAGE_ID_DIGEST_INVALID"
            )
        )
        image_groups.add(
            require_count(
                image["project_group_index"],
                0,
                len(groups) - 1,
                "GROUP_INDEX_OUT_OF_RANGE",
            )
        )
    if image_ids != sorted(set(image_ids)) or image_groups != set(range(len(groups))):
        hold("IMAGES_NOT_UNIQUE_SORTED_OR_GROUP_COVERAGE")
    paths = set()
    declared = []
    for f in files:
        require_keys(f, MANIFEST_FILE_KEYS, "PACKAGE_FILE_SCHEMA")
        relative = str(safe_relative(f["path"]))
        if relative in paths:
            hold("DUPLICATE_PACKAGE_FILE")
        paths.add(relative)
        require_count(f["size_bytes"], 1, MAX_MEMBER, "PACKAGE_MEMBER_SIZE_LIMIT")
        require_sha(f["sha256"], HEX64, "PACKAGE_MEMBER_DIGEST_INVALID")
        declared.append(relative)
    if declared != sorted(declared):
        hold("PACKAGE_FILE_ORDER_NOT_CANONICAL")
    required = set()
    per_image = [0] * len(images)
    for index, artifact in enumerate(artifacts):
        require_keys(artifact, {"image_index", "representation"}, "ARTIFACT_SCHEMA")
        image_index = require_count(
            artifact["image_index"], 0, len(images) - 1, "ARTIFACT_IMAGE_INDEX"
        )
        rep = artifact["representation"]
        if rep not in ARTIFACT_REP:
            hold("ARTIFACT_REP_UNSUPPORTED")
        per_image[image_index] += 1
        required.add(f"artifacts/{index}/origin.json")
        required.add(f"artifacts/{index}/{ARTIFACT_REP[rep]}")
    if [artifact["image_index"] for artifact in artifacts] != sorted(
        artifact["image_index"] for artifact in artifacts
    ):
        hold("ARTIFACTS_NOT_SORTED_BY_IMAGE")
    per_roi = [0] * len(images)
    roi_ids = set()
    prev_roi_key = None
    for index, roi in enumerate(rois):
        require_keys(
            roi, {"source_roi_uuid", "image_index", "source_envelope"}, "ROI_SCHEMA"
        )
        image_index = require_count(
            roi["image_index"], 0, len(images) - 1, "ROI_IMAGE_INDEX"
        )
        rid = roi["source_roi_uuid"]
        try:
            if str(uuid.UUID(rid)) != rid:
                hold("NONCANONICAL_ROI_UUID")
        except (TypeError, ValueError, AttributeError):
            hold("NONCANONICAL_ROI_UUID")
        key = (image_index, rid)
        if key in roi_ids or (prev_roi_key is not None and key < prev_roi_key):
            hold("ROI_ORDER_OR_DUPLICATE")
        roi_ids.add(key)
        prev_roi_key = key
        env = require_keys(
            roi["source_envelope"],
            {"artifact_index", "series_index", "resolution_level", "pixel_rect"},
            "SOURCE_ENVELOPE_SCHEMA",
        )
        a = require_count(
            env["artifact_index"], 0, len(artifacts) - 1, "ROI_ARTIFACT_INDEX"
        )
        if artifacts[a]["image_index"] != image_index:
            hold("CROSS_IMAGE_ARTIFACT_REFERENCE")
        for field in ("series_index", "resolution_level"):
            require_count(env[field], 0, 2**31 - 1, "SOURCE_LEVEL_INVALID")
        rect = require_keys(
            env["pixel_rect"], {"x", "y", "width", "height"}, "PIXEL_RECT_SCHEMA"
        )
        for field in ("x", "y"):
            require_count(rect[field], 0, 2**31 - 1, "PIXEL_COORD_INVALID")
        for field in ("width", "height"):
            require_count(rect[field], 1, 2**31 - 1, "PIXEL_SIZE_INVALID")
        per_roi[image_index] += 1
        required.add(f"rois/{index}/roi.geojson")
        required.add(f"rois/{index}/objects.geojson")
    if not all(per_image) or not all(per_roi) or paths != required:
        hold("INCOMPLETE_HISTOFORGE_PACKAGE_GRAPH")
    for image_index in range(len(images)):
        indexes = [
            i for i, a in enumerate(artifacts) if a["image_index"] == image_index
        ]
        representations = {artifacts[i]["representation"] for i in indexes}
        expected_rois = [r for r in rois if r["image_index"] == image_index]
        if "original" in representations and (
            representations != {"original"}
            or len(indexes) != 1
            or any(
                r["source_envelope"]["artifact_index"] != indexes[0]
                for r in expected_rois
            )
        ):
            hold("ORIGINAL_ARTIFACT_GRAPH_INVALID")
        if "original" not in representations:
            if len(indexes) != len(expected_rois):
                hold("DERIVED_ARTIFACT_COUNT_MISMATCH")
            if any(
                roi["source_envelope"]["artifact_index"] != artifact_idx
                for roi, artifact_idx in zip(expected_rois, indexes, strict=True)
            ):
                hold("DERIVED_ARTIFACT_PAIR_MISMATCH")
    return m, paths


def validate_split(value: Any, dataset: dict) -> dict:
    sidecar = require_keys(value, SPLIT_KEYS, "SPLIT_SIDECAR_SCHEMA")
    if type(sidecar["schema_version"]) is not int or sidecar["schema_version"] != 1:
        hold("SPLIT_SIDECAR_VERSION")
    if sidecar["dataset_digest"] != dataset["dataset_digest"]:
        hold("SPLIT_DATASET_DIGEST_MISMATCH")
    plan = sidecar["split_plan"]
    required = (
        SPLIT_PLAN_GENERATED_KEYS
        if isinstance(plan, dict) and plan.get("assignment_source") == "generated"
        else SPLIT_PLAN_KEYS
    )
    require_keys(plan, required, "SPLIT_PLAN_SCHEMA")
    if plan["type"] != "split" or plan["unit"] != "project_group":
        hold("SPECIMEN_LEVEL_SPLIT_REQUIRED")
    if plan["assignment_source"] not in {"generated", "custom"}:
        hold("SPLIT_ASSIGNMENT_SOURCE_UNSUPPORTED")
    if "requested_validation_percent" in plan:
        require_count(
            plan["requested_validation_percent"], 1, 99, "INVALID_VAL_PERCENT"
        )
    units = plan["units"]
    if not isinstance(units, list) or len(units) != len(dataset["project_groups"]):
        hold("SPLIT_UNIT_COVERAGE_MISMATCH")
    partitions = []
    for index, assignment in enumerate(units):
        require_keys(assignment, {"unit_index", "partition"}, "SPLIT_UNIT_SCHEMA")
        if (
            type(assignment["unit_index"]) is not int
            or assignment["unit_index"] != index
        ):
            hold("SPLIT_ASSIGNMENT_ORDER")
        if assignment["partition"] not in {"training", "validation"}:
            hold("INVALID_SPLIT_PARTITION")
        partitions.append(assignment["partition"])
    if set(partitions) != {"training", "validation"}:
        hold("TRAIN_AND_VALIDATION_REQUIRED")
    image_partition = [
        partitions[image["project_group_index"]] for image in dataset["images"]
    ]
    counts = {
        name: sum(x == name for x in image_partition)
        for name in ("training", "validation")
    }
    roi_counts = {
        name: sum(
            image_partition[roi["image_index"]] == name for roi in dataset["rois"]
        )
        for name in ("training", "validation")
    }
    if not all(counts.values()) or not all(roi_counts.values()):
        hold("PARTITION_IMAGE_OR_ROI_EMPTY")
    return {
        "image_count": counts,
        "roi_count": roi_counts,
        "group_count": {name: partitions.count(name) for name in counts},
    }


def verify_archive(archive: Path, manifest_limit: int = MAX_JSON) -> tuple[dict, dict]:
    try:
        with zipfile.ZipFile(archive, "r", allowZip64=True) as package:
            entries = package.infolist()
            # Never call testzip() before validating metadata limits: it
            # decompresses every member and could expand a ZIP bomb before
            # the declared size/ratio budgets are checked. Streamed member
            # reads below verify both SHA-256 and ZIP CRC after caps pass.
            if not 2 <= len(entries) <= MAX_MEMBERS:
                hold("ZIP_MEMBER_STRUCTURE_OR_CRC")
            names = []
            sizes = {}
            total = 0
            for info in entries:
                name = str(safe_relative(info.filename))
                if (
                    info.is_dir()
                    or info.flag_bits & 1
                    or info.compress_type
                    not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
                ):
                    hold("UNSUPPORTED_ZIP_MEMBER")
                if stat.S_IFMT(info.external_attr >> 16) == stat.S_IFLNK:
                    hold("ZIP_SYMLINK_FORBIDDEN")
                if not 0 < info.file_size <= MAX_MEMBER:
                    hold("ZIP_MEMBER_SIZE_LIMIT")
                if (
                    info.compress_size == 0
                    or info.file_size > info.compress_size * 1000
                ):
                    hold("ZIP_DECOMPRESSION_RATIO_LIMIT")
                total += info.file_size
                if total > MAX_TOTAL_UNCOMPRESSED:
                    hold("ZIP_TOTAL_EXPANSION_LIMIT")
                names.append(name)
                sizes[name] = info.file_size
            if len(names) != len(set(names)) or "dataset.json" not in names:
                hold("ZIP_DUPLICATE_OR_MISSING_MANIFEST")
            if sizes["dataset.json"] > manifest_limit:
                hold("DATASET_MANIFEST_LIMIT")
            raw = package.read("dataset.json")
            if len(raw) > manifest_limit:
                hold("DATASET_MANIFEST_LIMIT")
            manifest = safe_json(raw, manifest_limit)
            if raw != jcs(manifest):
                hold("NONCANONICAL_HISTOFORGE_DATASET_JSON")
            m, declared = manifest_checks(manifest)
            if set(names) != declared | {"dataset.json"}:
                hold("ZIP_MANIFEST_CLOSURE_MISMATCH")
            file_records = {entry["path"]: entry for entry in m["files"]}
            if sum(x["size_bytes"] for x in file_records.values()) + len(raw) != total:
                hold("ZIP_MANIFEST_TOTAL_SIZE_MISMATCH")
            for name, expected in file_records.items():
                if sizes[name] != expected["size_bytes"]:
                    hold("ZIP_ENTRY_BYTE_COUNT_MISMATCH")
                h = hashlib.sha256()
                with package.open(name, "r") as stream:
                    while block := stream.read(BLOCK_SIZE):
                        h.update(block)
                if h.hexdigest() != expected["sha256"]:
                    hold("ZIP_ENTRY_SHA256_MISMATCH")
            return m, {
                "file_count": len(declared),
                "total_uncompressed_bytes": total,
                "manifest_json_sha256": sha_bytes(raw),
            }
    except (zipfile.BadZipFile, zipfile.LargeZipFile, EOFError, OSError):
        hold("ZIP_READ_OR_INTEGRITY_FAILED")


def inspect_intake(request_path: Path) -> dict[str, Any]:
    root = request_path.resolve().parent
    source = safe_json(read_bounded(request_path, MAX_REQUEST), MAX_REQUEST)
    request = require_keys(source, REQUIRED_REQUEST, "INTAKE_REQUEST_SCHEMA")
    if request["schema"] != INTAKE_SCHEMA:
        hold("INTAKE_SCHEMA_VERSION")
    upstream = require_keys(request["upstream"], UPSTREAM_KEYS, "UPSTREAM_SCHEMA")
    required_commits = {
        "backend_commit": UPSTREAM_BACKEND_SHA,
        "models_commit": UPSTREAM_MODELS_SHA,
        "extension_commit": UPSTREAM_EXTENSION_SHA,
    }
    if upstream != required_commits:
        hold("HISTOFORGE_SOURCE_DRIFT_REVIEW_REQUIRED")
    selected = require_keys(request["module"], MODULE_KEYS, "MODULE_SCHEMA")
    if (
        not isinstance(selected["repo"], str)
        or not REPO.fullmatch(selected["repo"])
        or not selected["repo"].lower().startswith("histoforge/")
    ):
        hold("MODULE_REPO_NOT_ADMITTED")
    require_sha(selected["revision"], HEX40, "UNPINNED_MODULE")
    if not isinstance(selected["model_id"], str) or not re.fullmatch(
        r"[a-z][a-z0-9_]{0,47}", selected["model_id"]
    ):
        hold("INVALID_MODEL_ID")
    if selected["weight_terms"] != "UNVERIFIED":
        # A string claim cannot confer rights; keep this adapter strictly
        # review-only even if a caller supplies an arbitrary license name.
        hold("MODEL_LICENSE_NOT_INDEPENDENTLY_VERIFIED")
    permissions = require_keys(
        request["permissions"], PERMISSIONS_KEYS, "PERMISSIONS_SCHEMA"
    )
    if any(type(v) is not bool for v in permissions.values()):
        hold("PERMISSIONS_VALUES_INVALID")
    archive = verify_input_file(root, request["archive"])
    sidecar_path = verify_input_file(root, request["split_sidecar"])
    if archive == sidecar_path:
        hold("SOURCE_AND_SPLIT_MUST_BE_DISTINCT")
    dataset, archive_info = verify_archive(archive)
    sidecar_raw = read_bounded(sidecar_path, MAX_JSON)
    split = safe_json(sidecar_raw, MAX_JSON)
    if sidecar_raw != jcs(split):
        hold("NONCANONICAL_HISTOFORGE_SPLIT")
    split_counts = validate_split(split, dataset)
    result = {
        "schema": RESULT_SCHEMA,
        "status": "SOURCE_BYTES_AND_SPECIMEN_SPLIT_CHECKED",
        "claim": "OFFLINE_TRAINING_INTAKE_ONLY",
        "training_execution_authorized": False,
        "model_inference_qualified": False,
        "clinical_diagnostic_qualified": False,
        "upload_authorized": False,
        "publication_authorized": False,
        "source_provenance_signed": False,
        "upstream": required_commits,
        "model_ref_binding_sha256": sha_bytes(
            jcs(
                {
                    "model_id": selected["model_id"],
                    "repo": selected["repo"],
                    "revision": selected["revision"],
                }
            )
        ),
        "data": {
            "archive_sha256": request["archive"]["sha256"],
            "split_sidecar_sha256": request["split_sidecar"]["sha256"],
            "histo_dataset_digest": dataset["dataset_digest"],
            "image_count": len(dataset["images"]),
            "roi_count": len(dataset["rois"]),
            "project_group_count": len(dataset["project_groups"]),
            **archive_info,
            "split_counts": split_counts,
        },
        "permission_declarations_are_independent_proof": False,
        "next_gates": [
            "UPSTREAM_HISTOFORGE_FULL_SCIENTIFIC_DATASET_VALIDATOR",
            "SOURCE_IMAGE_PIXELS_CALIBRATION_AND_ROI_LABEL_GEOMETRY",
            "SPLIT_SPECIMEN_IDENTITIES_INDEPENDENTLY_REVIEWED",
            "DATA_USE_AND_TRANSFER_PERMISSIONS_INDEPENDENTLY_REVIEWED",
            "GATED_MODEL_WEIGHT_TERMS_AND_RIGHTS_INDEPENDENTLY_REVIEWED",
            "HELD_OUT_INDEPENDENT_TEST_SET_AND_BASELINE",
            "SIGNED_OWNER_GPU_JOB_CAPACITY_BUDGET_AND_QUEUE_ADMISSION",
            "FINAL_ADAPTER_AND_EVALUATION_RECEIPTS_BEFORE_HF_PUBLICATION",
        ],
        "limits": "pilot ZIP <=128 MiB; no slide, artifact or source code executed",
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        evidence = inspect_intake(args.request)
        payload = (
            json.dumps(evidence, sort_keys=True, indent=2, ensure_ascii=True) + "\n"
        ).encode()
        # Exclusive create; never clobber an operator record or source file.
        if args.out.is_symlink():
            hold("OUTPUT_SYMLINK_FORBIDDEN")
        with args.out.open("xb") as dest:
            dest.write(payload)
            dest.flush()
            os.fsync(dest.fileno())
        print(
            json.dumps(
                {
                    "status": evidence["status"],
                    "training_started": False,
                    "evidence_sha256": sha_bytes(payload),
                },
                sort_keys=True,
            )
        )
        return 0
    except IntakeHold as exc:
        print(
            json.dumps(
                {"status": "HOLD", "reason_code": str(exc), "training_started": False},
                sort_keys=True,
            )
        )
        return 2
    except (OSError, FileExistsError):
        print(
            json.dumps(
                {
                    "status": "HOLD",
                    "reason_code": "FILE_OR_OUTPUT_UNAVAILABLE",
                    "training_started": False,
                },
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())
