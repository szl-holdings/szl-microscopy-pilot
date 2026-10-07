"""Deterministic, CPU-only quality measurements for a bounded image pair.

The output describes pixels and acquisition provenance. It does not interpret
biology, choose data exclusions, or perform atlas/model calls.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from . import __version__


MAX_INPUT_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 4_000_000
MIN_PIXEL_VARIANCE = 1e-10
SHA256 = re.compile(r"^[0-9a-f]{64}$")
IMAGE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


class PilotError(ValueError):
    """A manifest, provenance, or image failed a required gate."""


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PilotError(f"{name} must be a nonempty string")
    return value.strip()


def _https_url(value: Any, name: str) -> str:
    value = _required_text(value, name)
    if not value.startswith("https://"):
        raise PilotError(f"{name} must be an HTTPS URL")
    return value


def _load_manifest(manifest_path: Path) -> tuple[dict[str, Any], Path]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PilotError(f"cannot read manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise PilotError("manifest must be a JSON object")
    allowed = {"dataset", "images", "claim", "atlas", "specialist_model", "comparison"}
    unknown = set(manifest) - allowed
    if unknown:
        raise PilotError(f"unsupported manifest fields: {', '.join(sorted(unknown))}")
    if manifest.get("claim") != "observable_quality_only":
        raise PilotError("only observable_quality_only claims are supported")
    if manifest.get("atlas") != {"configured": False}:
        raise PilotError("atlas matching is not configured; atlas evidence cannot be claimed")
    if manifest.get("specialist_model", {"configured": False}) != {"configured": False}:
        raise PilotError("specialist models are not configured")
    dataset = manifest.get("dataset")
    if not isinstance(dataset, dict):
        raise PilotError("dataset metadata is required")
    if set(dataset) != {"accession", "version", "citation", "source_page", "license_url"}:
        raise PilotError("dataset requires accession, version, citation, source_page, license_url")
    for key in ("accession", "version", "citation"):
        _required_text(dataset[key], f"dataset.{key}")
    for key in ("source_page", "license_url"):
        _https_url(dataset[key], f"dataset.{key}")
    images = manifest.get("images")
    if not isinstance(images, list) or not 1 <= len(images) <= 2:
        raise PilotError("images must contain one or two bounded inputs")
    comparison = manifest.get("comparison", {"requested": False})
    if comparison not in ({"requested": True}, {"requested": False}):
        raise PilotError("comparison must contain only a boolean requested flag")
    return manifest, manifest_path.resolve().parent


def _read_verified_image(item: dict[str, Any], root: Path) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise PilotError("each image entry must be an object")
    allowed = {
        "id", "path", "source_url", "source_member", "source_byte_range", "sha256", "bytes",
        "label", "field_id", "channel", "z_plane", "exposure_ms", "gain", "processing",
    }
    unknown = set(item) - allowed
    if unknown:
        raise PilotError(f"unsupported image fields: {', '.join(sorted(unknown))}")
    for key in ("id", "path", "label"):
        _required_text(item.get(key), f"image.{key}")
    if not IMAGE_ID.fullmatch(item["id"]):
        raise PilotError("image.id must use letters, digits, underscore, or hyphen")
    _https_url(item.get("source_url"), "image.source_url")
    digest = item.get("sha256")
    if not isinstance(digest, str) or not SHA256.fullmatch(digest):
        raise PilotError("image.sha256 must be a lowercase SHA-256 hex digest")
    expected_bytes = item.get("bytes")
    if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int) or not 0 < expected_bytes <= MAX_INPUT_BYTES:
        raise PilotError("image.bytes must be within the 8 MiB input limit")
    path = (root / item["path"]).resolve()
    if not path.is_relative_to(root):
        raise PilotError("image path must stay inside the manifest directory")
    try:
        if path.stat().st_size != expected_bytes:
            raise PilotError(f"byte count mismatch for {item['id']}")
        data = path.read_bytes()
    except OSError as exc:
        raise PilotError(f"cannot read image {item['id']}: {exc}") from exc
    if len(data) != expected_bytes:
        raise PilotError(f"byte count mismatch for {item['id']}")
    if hashlib.sha256(data).hexdigest() != digest:
        raise PilotError(f"SHA-256 mismatch for {item['id']}")
    if "source_member" in item:
        _required_text(item["source_member"], "image.source_member")
    if "source_byte_range" in item:
        byte_range = item["source_byte_range"]
        if (not isinstance(byte_range, list) or len(byte_range) != 2
                or any(isinstance(x, bool) or not isinstance(x, int) for x in byte_range)
                or byte_range[0] < 0 or byte_range[1] < byte_range[0]):
            raise PilotError("image.source_byte_range must be [start, end] inclusive")
    for key in ("field_id", "channel", "z_plane", "processing"):
        if key in item:
            _required_text(item[key], f"image.{key}")
    for key in ("exposure_ms", "gain"):
        if key in item and (isinstance(item[key], bool) or not isinstance(item[key], (int, float))
                            or not math.isfinite(item[key]) or item[key] <= 0):
            raise PilotError(f"image.{key} must be a positive finite number")
    metrics = _measure_image(path)
    result = {key: item[key] for key in (
        "id", "label", "source_url", "source_member", "source_byte_range", "sha256", "bytes",
        "field_id", "channel", "z_plane", "exposure_ms", "gain", "processing",
    ) if key in item}
    result["metrics"] = metrics
    return result


def _measure_image(path: Path) -> dict[str, Any]:
    try:
        import PIL
        from PIL import Image, UnidentifiedImageError
    except ImportError as exc:
        raise PilotError("Pillow==12.3.0 is required; install the pinned dependency") from exc
    if PIL.__version__ != "12.3.0":
        raise PilotError("the reproducible run requires Pillow==12.3.0")
    try:
        with Image.open(path) as im:
            if im.format not in {"PNG", "TIFF"} or getattr(im, "n_frames", 1) != 1:
                raise PilotError("only single-frame PNG or TIFF images are supported")
            width, height = im.size
            if width < 3 or height < 3 or width * height > MAX_PIXELS:
                raise PilotError("image dimensions exceed the bounded QC limits")
            if im.mode in {"I;16", "I;16L", "I;16B"}:
                bit_depth = 16
                pixels = [p / 65535.0 for p in im.getdata()]
            elif im.mode == "L":
                bit_depth = 8
                pixels = [p / 255.0 for p in im.getdata()]
            elif im.mode == "RGB":
                bit_depth = 8
                pixels = [(2126 * r + 7152 * g + 722 * b) / 2_550_000.0
                          for r, g, b in im.getdata()]
            else:
                raise PilotError(f"unsupported image mode: {im.mode}")
    except PilotError:
        raise
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise PilotError(f"cannot decode image {path.name}: {exc}") from exc
    count = len(pixels)
    mean = sum(pixels) / count
    pixel_variance = sum((p - mean) ** 2 for p in pixels) / count
    laps = []
    for y in range(1, height - 1):
        row = y * width
        for x in range(1, width - 1):
            i = row + x
            laps.append(4 * pixels[i] - pixels[i - 1] - pixels[i + 1]
                        - pixels[i - width] - pixels[i + width])
    lap_mean = sum(laps) / len(laps)
    lap_energy = sum(v * v for v in laps) / len(laps)
    lap_variance = lap_energy - lap_mean * lap_mean
    focus_proxy = lap_energy / (mean * mean) if mean > 0 else 0.0
    endpoint = 1 / (65535 if bit_depth == 16 else 255)
    return {
        "width": width,
        "height": height,
        "bit_depth": bit_depth,
        "mean_intensity": round(mean, 12),
        "pixel_variance": round(pixel_variance, 12),
        "clipped_dark_fraction": round(sum(p <= endpoint for p in pixels) / count, 12),
        "clipped_bright_fraction": round(sum(p >= 1 - endpoint for p in pixels) / count, 12),
        "laplacian_variance": round(max(0.0, lap_variance), 12),
        "focus_proxy": round(focus_proxy, 12),
    }


def _comparison(images: list[dict[str, Any]], requested: bool) -> dict[str, Any]:
    result: dict[str, Any] = {"status": "not_requested", "reason": "Pair ranking was not requested.",
                              "higher_focus_proxy_image_id": None}
    if not requested:
        return result
    result["status"] = "not_comparable"
    if len(images) != 2:
        result["reason"] = "Pair ranking requires exactly two images."
        return result
    a, b = images
    if not all(a.get(k) and a.get(k) == b.get(k) for k in ("field_id", "channel")):
        result["reason"] = "Field ID and channel must be present and equal."
        return result
    if (a["metrics"]["width"], a["metrics"]["height"]) != (b["metrics"]["width"], b["metrics"]["height"]):
        result["reason"] = "Image dimensions differ."
        return result
    for key in ("exposure_ms", "gain", "processing"):
        if (key in a or key in b) and (key not in a or key not in b or a[key] != b[key]):
            result["reason"] = f"{key} differs or is missing for one image."
            return result
    if any(i["metrics"]["pixel_variance"] < MIN_PIXEL_VARIANCE for i in images):
        result["status"] = "insufficient_texture"
        result["reason"] = "At least one image has insufficient measured texture for this proxy."
        return result
    if a["metrics"]["focus_proxy"] == b["metrics"]["focus_proxy"]:
        result["reason"] = "The measured focus proxies are tied."
        return result
    higher = max(images, key=lambda i: i["metrics"]["focus_proxy"])
    result["status"] = "proxy_ranked"
    result["higher_focus_proxy_image_id"] = higher["id"]
    result["reason"] = (
        "Matched field, channel, and dimensions support a relative proxy ranking. "
        "Exposure/gain equality is not independently verified when absent from source metadata."
    )
    return result


def run(manifest_path: Path, output_path: Path | None = None) -> dict[str, Any]:
    """Validate source bytes, measure pixels, and optionally write a stable JSON report."""
    manifest_path = Path(manifest_path)
    manifest, root = _load_manifest(manifest_path)
    images = [_read_verified_image(item, root) for item in manifest["images"]]
    if len({image["id"] for image in images}) != len(images):
        raise PilotError("image IDs must be unique")
    comparison = _comparison(images, manifest.get("comparison", {"requested": False})["requested"])
    if comparison["status"] == "proxy_ranked":
        higher = next(i for i in images if i["id"] == comparison["higher_focus_proxy_image_id"])
        lower = next(i for i in images if i["id"] != higher["id"])
        claim = (f"For the supplied matched field and channel, {higher['id']} has a higher measured "
                 f"focus proxy than {lower['id']}; this is a relative image-quality observation.")
    else:
        claim = "Quality metrics were measured; a comparable focus-proxy ranking is not supported."
    limitations = [
        "The focus proxy can increase with deterministic noise, sharpening, contrast, or gain changes.",
        "An encoded 16-bit TIFF range does not establish detector saturation or biological signal range.",
        "No biological meaning, exclusion decision, atlas match, or specialist-model result is established.",
        "Source URLs and acquisition metadata come from the manifest; this run verifies local bytes, not remote provenance.",
    ]
    evidence = [{"image_id": image["id"], "metric": "focus_proxy",
                 "value": image["metrics"]["focus_proxy"],
                 "pointer": f"/images/{index}/metrics/focus_proxy", "source_url": image["source_url"]}
                for index, image in enumerate(images)]
    report: dict[str, Any] = {
        "schema_version": "1.0.0",
        "dataset": manifest["dataset"],
        "method": {
            "name": "mean-normalized squared 4-neighbor Laplacian energy",
            "version": __version__,
            "parameters": {"interior_only": True, "min_pixel_variance": MIN_PIXEL_VARIANCE,
                           "max_input_bytes": MAX_INPUT_BYTES, "max_pixels": MAX_PIXELS,
                           "decoder": "Pillow 12.3.0"},
            "limitations": limitations,
        },
        "images": images,
        "interpretation_scope": "observable_image_quality",
        "comparison": comparison,
        "explanation": {
            "claim": claim,
            "claim_type": "observable_image_quality",
            "supporting_evidence": evidence,
            "counter_evidence": [
                {"observation": "Noise and processing can raise this proxy without improved focus.",
                 "pointer": "/method/limitations/0"},
                {"observation": "No biological interpretation follows from image sharpness.",
                 "pointer": "/method/limitations/2"},
                {"observation": "Source metadata is supplied by the manifest rather than checked against the archive at run time.",
                 "pointer": "/method/limitations/3"},
            ],
            "uncertainty": {"level": "limited", "reason": comparison["reason"]},
        },
        "human_review_required": True,
        "human_decision": {"status": "pending", "options": ["keep", "exclude", "reacquire", "uncertain"],
                           "rationale": None},
        "biological_assessment": {"status": "abstain", "reason": "No independent biological evidence was supplied."},
        "atlas": {"status": "not_configured", "evidence": []},
        "specialist_model": {"status": "not_configured", "evidence": []},
    }
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(output_path.name + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(output_path)
    return report
