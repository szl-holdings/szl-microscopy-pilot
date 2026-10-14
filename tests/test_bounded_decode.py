"""Offline decoder admission and bounded pixel-access regressions.

Every image here is a synthetic parser input generated at test time. None is a
BBBC image, a biological sample, or a reference focus annotation. The tests
pin behaviour, not memory use: no RSS or out-of-memory result is measured.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

from PIL import Image, PngImagePlugin, TiffImagePlugin

from microscopy_pilot.core import (
    MAX_INPUT_BYTES,
    MAX_PIXELS,
    PilotError,
    _measure_image,
    run,
)


ROOT = Path(__file__).resolve().parents[1]
NATIVE_MANIFEST = ROOT / "examples/bbbc006v1/manifest.json"
DECODE_REACHED = "pixel decode reached"
# Calls that build a full-image raw pixel sequence (or are deprecated).
EAGER_PIXEL_CALLS = ("getdata", "get_flattened_data", "tobytes")
IMAGE_CLASSES = (PngImagePlugin.PngImageFile, TiffImagePlugin.TiffImageFile)


def _forbid_pixel_decode() -> contextlib.ExitStack:
    """Make every load() in the PNG/TIFF class hierarchies raise."""
    stack = contextlib.ExitStack()
    patched: set[type] = set()
    for image_class in IMAGE_CLASSES:
        for klass in image_class.__mro__:
            if "load" in vars(klass) and klass not in patched:
                patched.add(klass)
                stack.enter_context(patch.object(
                    klass, "load", side_effect=AssertionError(DECODE_REACHED)))
    return stack


def _forbid_eager_pixel_calls() -> contextlib.ExitStack:
    stack = contextlib.ExitStack()
    for name in EAGER_PIXEL_CALLS:
        stack.enter_context(patch.object(
            Image.Image, name, side_effect=AssertionError(f"eager pixel call: {name}")))
    return stack


@contextlib.contextmanager
def _record_pixel_access():
    """Wrap the PixelAccess returned by load() and record each coordinate read."""
    visits: list[tuple[int, int]] = []

    class RecordingAccess:
        def __init__(self, access):
            self._access = access

        def __getitem__(self, xy):
            visits.append(tuple(xy))
            return self._access[xy]

    with contextlib.ExitStack() as stack:
        for image_class in IMAGE_CLASSES:
            original = image_class.load

            def load(self, _original=original):
                return RecordingAccess(_original(self))

            stack.enter_context(patch.object(image_class, "load", load))
        yield visits


def _row_major(width: int, height: int) -> list[tuple[int, int]]:
    return [(x, y) for y in range(height) for x in range(width)]


def _first_divergence(actual: list, expected: list) -> str | None:
    """Cheap mismatch report; assertEqual's diff is quadratic on ~720k visits."""
    if actual == expected:
        return None
    index = next((i for i, (a, e) in enumerate(zip(actual, expected)) if a != e),
                 min(len(actual), len(expected)))
    return (f"{len(actual)} visits, expected {len(expected)}; first divergence at "
            f"{index}: {actual[index:index + 3]} != {expected[index:index + 3]}")


def _reference_metrics(pixels: list[float], width: int, height: int, bit_depth: int) -> dict:
    """Documented QC formulas applied to independently known pixel values."""
    count = len(pixels)
    mean = sum(pixels) / count
    pixel_variance = sum((p - mean) ** 2 for p in pixels) / count
    laps = [
        4 * pixels[y * width + x] - pixels[y * width + x - 1] - pixels[y * width + x + 1]
        - pixels[(y - 1) * width + x] - pixels[(y + 1) * width + x]
        for y in range(1, height - 1) for x in range(1, width - 1)
    ]
    lap_mean = sum(laps) / len(laps)
    lap_energy = sum(v * v for v in laps) / len(laps)
    endpoint = 1 / (65535 if bit_depth == 16 else 255)
    return {
        "width": width,
        "height": height,
        "bit_depth": bit_depth,
        "mean_intensity": round(mean, 12),
        "pixel_variance": round(pixel_variance, 12),
        "clipped_dark_fraction": round(sum(p <= endpoint for p in pixels) / count, 12),
        "clipped_bright_fraction": round(sum(p >= 1 - endpoint for p in pixels) / count, 12),
        "laplacian_variance": round(max(0.0, lap_energy - lap_mean * lap_mean), 12),
        "focus_proxy": round(lap_energy / (mean * mean), 12),
    }


class DecodeAdmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def _save(self, name: str, mode: str, size: tuple[int, int], fill=0) -> Path:
        path = self.root / name
        Image.new(mode, size, fill).save(path)
        return path

    def test_unsupported_modes_are_rejected_before_any_pixel_decode(self) -> None:
        # Negative fixtures: mode is known from the header, so an unsupported
        # mode must be refused without decoding the pixel data first.
        fixtures = [
            ("rgba.png", "RGBA", (1, 2, 3, 4)),
            ("la.png", "LA", (1, 2)),
            ("palette.png", "P", 3),
            ("bilevel.png", "1", 1),
            ("float.tif", "F", 1.5),
            ("int32.tif", "I", 70000),
        ]
        for name, mode, fill in fixtures:
            path = self._save(name, mode, (5, 4), fill)
            with self.subTest(mode=mode), _forbid_pixel_decode():
                with self.assertRaisesRegex(PilotError, f"unsupported image mode: {mode}$"):
                    _measure_image(path)

    def test_oversized_dimensions_are_rejected_before_any_pixel_decode(self) -> None:
        # Negative fixtures: one pixel over the four-million-pixel limit in each
        # orientation and supported mode, plus a too-small edge.
        fixtures = [
            ("wide-l.png", "L", (2001, 2000)),
            ("tall-rgb.png", "RGB", (2000, 2001)),
            ("wide-16.png", "I;16", (4001, 1000)),
            ("thin-l.png", "L", (2, 5)),
        ]
        for name, mode, size in fixtures:
            path = self._save(name, mode, size)
            with self.subTest(mode=mode, size=size), _forbid_pixel_decode():
                with self.assertRaisesRegex(PilotError, "exceed the bounded QC limits"):
                    _measure_image(path)

    def test_four_million_pixel_boundary_is_admitted_to_pixel_decode(self) -> None:
        # Passing control: exactly MAX_PIXELS clears every header gate, and the
        # next step is the pixel decode (stopped here to keep the test fast).
        self.assertEqual(MAX_PIXELS, 4_000_000)
        path = self._save("boundary.png", "L", (2000, 2000))
        with _forbid_pixel_decode():
            with self.assertRaisesRegex(AssertionError, DECODE_REACHED):
                _measure_image(path)

    def _manifest_with(self, data: bytes, declared_bytes: int) -> Path:
        image = self.root / "input.png"
        image.write_bytes(data)
        manifest = {
            "dataset": {
                "accession": "SYNTHETIC-TEST",
                "version": "1",
                "citation": "Synthetic test fixture generated in tests/test_bounded_decode.py",
                "source_page": "https://example.org/synthetic-test-fixture",
                "license_url": "https://example.org/synthetic-test-fixture-license",
            },
            "images": [{
                "id": "synthetic",
                "path": image.name,
                "source_url": "https://example.org/synthetic-test-fixture.png",
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": declared_bytes,
                "label": "synthetic parser fixture",
            }],
            "claim": "observable_quality_only",
            "atlas": {"configured": False},
        }
        path = self.root / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def test_eight_mib_encoded_input_limit_is_enforced_before_reading(self) -> None:
        self.assertEqual(MAX_INPUT_BYTES, 8 * 1024 * 1024)
        over = b"\0" * (MAX_INPUT_BYTES + 1)
        # Negative fixtures: an over-limit declaration, and an over-limit file
        # declared at the limit, are both refused before the bytes are read.
        cases = [
            (MAX_INPUT_BYTES + 1, "must be within the 8 MiB input limit"),
            (MAX_INPUT_BYTES, "byte count mismatch"),
        ]
        for declared, refusal in cases:
            manifest = self._manifest_with(over, declared)
            with self.subTest(declared=declared), patch.object(
                Path, "read_bytes", side_effect=AssertionError("encoded input read"),
            ), _forbid_pixel_decode():
                with self.assertRaisesRegex(PilotError, refusal):
                    run(manifest)

    def test_eight_mib_boundary_is_admitted_to_the_decoder(self) -> None:
        # Passing control: exactly MAX_INPUT_BYTES is read, hashed and handed to
        # the decoder, which then refuses the non-image bytes.
        manifest = self._manifest_with(b"\0" * MAX_INPUT_BYTES, MAX_INPUT_BYTES)
        with self.assertRaisesRegex(PilotError, "cannot decode image"):
            run(manifest)


class BoundedPixelAccessTests(unittest.TestCase):
    WIDTH, HEIGHT = 7, 5  # non-square, so a transposed walk changes results

    def _values(self) -> list[int]:
        values = [(x * x * 37 + y * 11 + (x ^ y) * 5) % 256
                  for y in range(self.HEIGHT) for x in range(self.WIDTH)]
        values[0], values[-1] = 0, 255  # exercise both encoded endpoints
        return values

    def _cases(self):
        values = self._values()
        rgb = [(v, (v * 7 + 13) % 256, 255 - v) for v in values]  # non-neutral
        wide = [v * 257 - (i % 3) if v else 0 for i, v in enumerate(values)]
        yield ("L", values, [v / 255.0 for v in values], 8)
        yield ("RGB", rgb, [(2126 * r + 7152 * g + 722 * b) / 2_550_000.0 for r, g, b in rgb], 8)
        yield ("I;16", wide, [v / 65535.0 for v in wide], 16)

    def test_supported_modes_read_each_coordinate_once_with_exact_arithmetic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for mode, raw, normalized, bit_depth in self._cases():
                path = Path(tmp) / f"{mode.replace(';', '_')}.png"
                image = Image.new(mode, (self.WIDTH, self.HEIGHT))
                image.putdata(raw)
                image.save(path)
                with self.subTest(mode=mode), warnings.catch_warnings(), \
                        _forbid_eager_pixel_calls(), _record_pixel_access() as visits:
                    warnings.simplefilter("error", DeprecationWarning)
                    metrics = _measure_image(path)
                    self.assertIsNone(_first_divergence(visits, _row_major(self.WIDTH, self.HEIGHT)))
                    self.assertEqual(
                        metrics, _reference_metrics(normalized, self.WIDTH, self.HEIGHT, bit_depth))

    def test_native_16_bit_tiff_reads_each_coordinate_once_in_row_major_order(self) -> None:
        with warnings.catch_warnings(), _forbid_eager_pixel_calls(), \
                _record_pixel_access() as visits:
            warnings.simplefilter("error", DeprecationWarning)
            report = run(NATIVE_MANIFEST)
        self.assertEqual(len(report["images"]), 2)
        width = report["images"][0]["metrics"]["width"]
        height = report["images"][0]["metrics"]["height"]
        self.assertIsNone(_first_divergence(visits, _row_major(width, height) * 2))


if __name__ == "__main__":
    unittest.main()
