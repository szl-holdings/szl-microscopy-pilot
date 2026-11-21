"""CPU-only, fail-closed multiscale volume coordinate/provenance witness.

Reference: OME-NGFF coordinateTransformation semantics, developed by the
Open Microscopy Environment (OME-Zarr, BSD-2-Clause). This is original SZL
code for a small, private **synthetic raw byte** fixture. It is NOT an OME-Zarr
reader/writer, does not establish OME-NGFF compliance, and does not render.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

MAX_METADATA_BYTES = 64 * 1024
MAX_LEVEL_BYTES = 4 * 1024 * 1024
MAX_TOTAL_BYTES = 8 * 1024 * 1024
MAX_LEVELS = 8
SHA = re.compile(r"^[0-9a-f]{64}$")
BASENAME = re.compile(r"^[A-Za-z0-9_-]{1,64}\.raw$")
AXES = ("z", "y", "x")


class GeometryError(ValueError):
    """A geometry, content, or provenance boundary was invalid."""


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GeometryError("DUPLICATE_JSON_KEY")
        result[key] = value
    return result


def _deny_nonfinite(_literal: str) -> None:
    raise GeometryError("NONFINITE_JSON_VALUE")


def _finite_float(literal: str) -> float:
    result = float(literal)
    if not math.isfinite(result):
        raise GeometryError("NONFINITE_JSON_VALUE")
    return result


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def read_manifest(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as fd:
            raw = fd.read(MAX_METADATA_BYTES + 1)
        if len(raw) > MAX_METADATA_BYTES:
            raise GeometryError("METADATA_TOO_LARGE")
        value = json.loads(raw, object_pairs_hook=_unique_pairs,
                           parse_constant=_deny_nonfinite, parse_float=_finite_float)
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        if isinstance(exc, GeometryError):
            raise
        raise GeometryError("INVALID_OR_UNREADABLE_METADATA_JSON") from None
    if not isinstance(value, dict):
        raise GeometryError("METADATA_NOT_OBJECT")
    return value


def _positive_shape(value: Any) -> tuple[int, int, int]:
    if (not isinstance(value, list) or len(value) != 3 or
            any(type(x) is not int or not 1 <= x <= 1024 for x in value)):
        raise GeometryError("INVALID_VOLUME_SHAPE")
    return tuple(value)


def _numbers(value: Any, *, positive: bool) -> tuple[float, float, float]:
    if (not isinstance(value, list) or len(value) != 3 or
            any(type(x) not in (int, float) or not math.isfinite(x) or
                (positive and x <= 0) for x in value)):
        raise GeometryError("INVALID_PHYSICAL_TRANSFORM")
    return tuple(float(x) for x in value)


def audit_multiscale_geometry(manifest: dict[str, Any], root: Path) -> dict[str, Any]:
    """Verify declared ZYX micrometre geometry + SHA of bounded raw bytes.

    Exact raw byte fidelity is tested; voxel semantics and correctness of the
    claimed physical scale require independent acquisition evidence.
    """
    if not isinstance(manifest, dict) or set(manifest) != {"schema", "identity", "axes", "levels"}:
        raise GeometryError("INVALID_MULTISCALE_MANIFEST_FIELDS")
    if manifest["schema"] != "szl.microscopy.synthetic-multiscale/v1":
        raise GeometryError("UNSUPPORTED_MULTISCALE_SCHEMA")
    identity = manifest["identity"]
    if not isinstance(identity, dict) or set(identity) != {"specimen_id", "channel", "source_revision"}:
        raise GeometryError("INVALID_IDENTITY")
    for key in identity:
        val = identity[key]
        if not isinstance(val, str) or not 1 <= len(val) <= 128 or any(ord(c) < 32 for c in val):
            raise GeometryError("INVALID_IDENTITY")
    axes = manifest["axes"]
    if axes != [
        {"name": axis, "type": "space", "unit": "micrometer"} for axis in AXES
    ]:
        raise GeometryError("AXIS_ORDER_OR_UNIT_MISMATCH")
    levels = manifest["levels"]
    if not isinstance(levels, list) or not 2 <= len(levels) <= MAX_LEVELS:
        raise GeometryError("INVALID_MULTISCALE_LEVELS")

    root = root.resolve()
    proof = []
    total_bytes = 0
    world_box = None
    previous_shape = None
    previous_scale = None
    seen_paths: set[str] = set()
    for index, level in enumerate(levels):
        if not isinstance(level, dict) or set(level) != {"path", "shape", "scale_um", "translation_um", "sha256"}:
            raise GeometryError("INVALID_LEVEL_FIELDS")
        name = level["path"]
        if not isinstance(name, str) or not BASENAME.fullmatch(name) or name in seen_paths:
            raise GeometryError("INVALID_OR_DUPLICATE_LEVEL_PATH")
        seen_paths.add(name)
        shape = _positive_shape(level["shape"])
        scale = _numbers(level["scale_um"], positive=True)
        translation = _numbers(level["translation_um"], positive=False)
        digest = level["sha256"]
        if not isinstance(digest, str) or not SHA.fullmatch(digest):
            raise GeometryError("INVALID_LEVEL_SHA")
        byte_count = math.prod(shape)
        total_bytes += byte_count
        if byte_count > MAX_LEVEL_BYTES or total_bytes > MAX_TOTAL_BYTES:
            raise GeometryError("VOLUME_BYTES_OUT_OF_BOUNDS")
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise GeometryError("LEVEL_PATH_OUTSIDE_ROOT")
        try:
            with path.open("rb") as stream:
                raw = stream.read(byte_count + 1)
        except OSError:
            raise GeometryError("UNREADABLE_LEVEL_DATA") from None
        if len(raw) != byte_count or hashlib.sha256(raw).hexdigest() != digest:
            raise GeometryError("LEVEL_DATA_MISMATCH")
        box = tuple((translation[i], translation[i] + shape[i] * scale[i]) for i in range(3))
        if world_box is None:
            world_box = box
        else:
            if previous_shape is None or previous_scale is None:
                raise GeometryError("INTERNAL_STATE")
            # Each level is an integer-grid downsample from its predecessor.
            if any(previous_shape[i] % shape[i] != 0 or previous_shape[i] <= shape[i] for i in range(3)):
                raise GeometryError("INCONSISTENT_DOWNSAMPLE_SHAPE")
            if any(not math.isclose(scale[i], previous_scale[i] * previous_shape[i] / shape[i], rel_tol=1e-12, abs_tol=1e-12)
                   for i in range(3)):
                raise GeometryError("INCONSISTENT_DOWNSAMPLE_SCALE")
            if any(not math.isclose(box[i][j], world_box[i][j], rel_tol=1e-12, abs_tol=1e-12)
                   for i in range(3) for j in (0, 1)):
                raise GeometryError("WORLD_BOUNDS_MISMATCH")
        proof.append({
            "path": name, "index": index, "sha256": digest,
            "shape": list(shape), "scale_um": list(scale),
            "translation_um": list(translation),
        })
        previous_shape, previous_scale = shape, scale

    result: dict[str, Any] = {
        "schema": "szl.microscopy.synthetic-multiscale-witness/v1",
        "status": "SYNTHETIC_COORDINATES_AND_BYTES_PASSED",
        "specimen_id": identity["specimen_id"],
        "channel": identity["channel"],
        "source_revision": identity["source_revision"],
        "source_identity_verified": False,
        "ngff_conformance_verified": False,
        "holographic_runtime_verified": False,
        "biological_interpretation": "ABSTAIN",
        "axis_order": list(AXES),
        "world_bounds_um": [[lo, hi] for lo, hi in world_box],
        "total_bytes": total_bytes,
        "levels": proof,
        "manifest_sha256": _sha(manifest),
    }
    result["witness_sha256"] = _sha(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = audit_multiscale_geometry(read_manifest(args.manifest), args.manifest.parent)
    except GeometryError as exc:
        print(json.dumps({"status": "DENY", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
