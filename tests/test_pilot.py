"""Offline contract tests using an intentionally synthetic PNG fixture.

The generated pixels are only a parser and QC test input. They are not a BBBC
image, a biological sample, or a reference focus annotation.
"""

from __future__ import annotations

import hashlib
import json
import struct
import tempfile
import unittest
import zlib
from pathlib import Path
from typing import Callable

from microscopy_pilot.core import PilotError, run


def _chunk(kind: bytes, data: bytes) -> bytes:
    body = kind + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


def _gray_png(
    width: int = 8,
    height: int = 8,
    pixel: Callable[[int, int], int] | None = None,
) -> bytes:
    """Encode a small, valid 8-bit grayscale PNG without image dependencies."""
    if pixel is None:
        pixel = lambda x, y: 32 if (x + y) % 2 else 224
    pixels = bytearray()
    for y in range(height):
        pixels.append(0)  # PNG filter: None
        for x in range(width):
            pixels.append(pixel(x, y))
    return b"".join(
        (
            b"\x89PNG\r\n\x1a\n",
            _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)),
            _chunk(b"IDAT", zlib.compress(bytes(pixels))),
            _chunk(b"IEND", b""),
        )
    )


class PilotRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.image_bytes = _gray_png()
        (self.root / "synthetic.png").write_bytes(self.image_bytes)
        self.manifest = {
            "dataset": {
                "accession": "SYNTHETIC-TEST",
                "version": "1",
                "citation": "Synthetic test fixture generated in tests/test_pilot.py",
                "source_page": "https://example.org/synthetic-test-fixture",
                "license_url": "https://example.org/synthetic-test-fixture-license",
            },
            "images": [
                {
                    "id": "synthetic",
                    "path": "synthetic.png",
                    "source_url": "https://example.org/synthetic-test-fixture.png",
                    "sha256": hashlib.sha256(self.image_bytes).hexdigest(),
                    "bytes": len(self.image_bytes),
                    "label": "synthetic parser fixture",
                }
            ],
            "claim": "observable_quality_only",
            "atlas": {"configured": False},
        }
        self.manifest_path = self.root / "manifest.json"
        self.output_path = self.root / "report.json"

    def _run(self) -> dict:
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        return run(self.manifest_path, self.output_path)

    def _add_image(self, image_id: str, png_bytes: bytes) -> None:
        image_name = f"{image_id}.png"
        (self.root / image_name).write_bytes(png_bytes)
        self.manifest["images"].append(
            {
                "id": image_id,
                "path": image_name,
                "source_url": f"https://example.org/{image_name}",
                "sha256": hashlib.sha256(png_bytes).hexdigest(),
                "bytes": len(png_bytes),
                "label": "synthetic parser fixture",
            }
        )

    def _request_comparison(self) -> None:
        self.manifest["comparison"] = {"requested": True}
        for image in self.manifest["images"]:
            image.update(
                {
                    "field_id": "synthetic-field-1",
                    "channel": "synthetic-gray",
                    "exposure_ms": 10,
                    "gain": 1,
                    "processing": "none",
                }
            )

    def test_report_is_deterministic_and_requires_human_review(self) -> None:
        first = self._run()
        on_disk = json.loads(self.output_path.read_text(encoding="utf-8"))
        self.assertEqual(first, on_disk)
        self.assertEqual(first["interpretation_scope"], "observable_image_quality")
        self.assertTrue(first["human_review_required"])
        self.assertEqual(first["atlas"]["status"], "not_configured")
        self.assertEqual(first["comparison"]["status"], "not_requested")
        self.assertEqual(first["biological_assessment"]["status"], "abstain")
        self.assertEqual(first["human_decision"]["status"], "pending")
        self.assertEqual(first["atlas"]["evidence"], [])
        self.assertEqual(first["specialist_model"], {"status": "not_configured", "evidence": []})
        self.assertEqual(len(first["images"]), 1)
        self.assertEqual(first["explanation"]["claim_type"], "observable_image_quality")
        self.assertTrue(first["explanation"]["counter_evidence"])
        self.assertTrue(first["explanation"]["uncertainty"]["reason"])
        # Check the enum and closed object that previously drifted from the
        # producer. This uses the shipped schema as data, without a validator.
        schema_path = Path(__file__).resolve().parents[1] / "schemas" / "explanation.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertIn(
            first["explanation"]["uncertainty"]["level"],
            schema["properties"]["explanation"]["properties"]["uncertainty"]["properties"]["level"]["enum"],
        )
        decision_schema = schema["properties"]["human_decision"]
        self.assertFalse(decision_schema["additionalProperties"])
        self.assertEqual(set(first["human_decision"]), set(decision_schema["properties"]))
        self.assertEqual(set(first["human_decision"]), set(decision_schema["required"]))
        for item in first["explanation"]["supporting_evidence"]:
            path = item["pointer"].split("/")[1:]
            measured = first
            for segment in path:
                measured = measured[int(segment)] if isinstance(measured, list) else measured[segment]
            self.assertEqual(item["value"], measured)
            self.assertEqual(item["source_url"], self.manifest["images"][0]["source_url"])

        second = self._run()
        self.assertEqual(first, second)

    def test_missing_required_dataset_metadata_is_rejected(self) -> None:
        del self.manifest["dataset"]["citation"]
        with self.assertRaisesRegex(PilotError, "dataset requires"):
            self._run()

    def test_mismatched_image_hash_is_rejected(self) -> None:
        self.manifest["images"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(PilotError, "SHA-256 mismatch"):
            self._run()

    def test_corrupt_png_is_rejected_even_with_matching_hash(self) -> None:
        corrupt = b"not a PNG image"
        (self.root / "synthetic.png").write_bytes(corrupt)
        self.manifest["images"][0]["sha256"] = hashlib.sha256(corrupt).hexdigest()
        self.manifest["images"][0]["bytes"] = len(corrupt)
        with self.assertRaisesRegex(PilotError, "cannot decode image"):
            self._run()

    def test_unsupported_biological_claim_is_rejected(self) -> None:
        self.manifest["claim"] = "disease_diagnosis"
        with self.assertRaisesRegex(PilotError, "observable_quality_only"):
            self._run()

    def test_atlas_claim_without_evidence_is_rejected(self) -> None:
        self.manifest["atlas"] = {"configured": True}
        with self.assertRaisesRegex(PilotError, "atlas matching is not configured"):
            self._run()

    def test_constant_images_abstain_from_focus_ranking(self) -> None:
        constant = _gray_png(pixel=lambda _x, _y: 128)
        (self.root / "synthetic.png").write_bytes(constant)
        self.manifest["images"][0]["sha256"] = hashlib.sha256(constant).hexdigest()
        self.manifest["images"][0]["bytes"] = len(constant)
        self._add_image("constant2", constant)
        self._request_comparison()

        report = self._run()
        self.assertEqual(report["comparison"]["status"], "insufficient_texture")
        self.assertIsNone(report["comparison"]["higher_focus_proxy_image_id"])
        self.assertTrue(report["human_review_required"])

    def test_mismatched_field_or_channel_blocks_comparison(self) -> None:
        for mismatched_key, changed_value in (
            ("field_id", "synthetic-field-2"),
            ("channel", "another-channel"),
        ):
            with self.subTest(mismatched_key=mismatched_key):
                self.manifest["images"] = self.manifest["images"][:1]
                self._add_image("second", _gray_png())
                self._request_comparison()
                self.manifest["images"][1][mismatched_key] = changed_value

                report = self._run()
                self.assertEqual(report["comparison"]["status"], "not_comparable")
                self.assertIsNone(report["comparison"]["higher_focus_proxy_image_id"])

    def test_gain_or_processing_change_blocks_comparison(self) -> None:
        for mismatched_key, changed_value in (
            ("gain", 2),
            ("processing", "normalized"),
        ):
            with self.subTest(mismatched_key=mismatched_key):
                self.manifest["images"] = self.manifest["images"][:1]
                self._add_image("second", _gray_png())
                self._request_comparison()
                self.manifest["images"][1][mismatched_key] = changed_value

                report = self._run()
                self.assertEqual(report["comparison"]["status"], "not_comparable")
                self.assertIsNone(report["comparison"]["higher_focus_proxy_image_id"])

    def test_different_dimensions_block_comparison(self) -> None:
        self._add_image("second", _gray_png(width=9, height=8))
        self._request_comparison()

        report = self._run()
        self.assertEqual(report["comparison"]["status"], "not_comparable")
        self.assertIsNone(report["comparison"]["higher_focus_proxy_image_id"])

    def test_missing_optional_acquisition_metadata_limits_proxy_ranking(self) -> None:
        self._add_image("second", _gray_png(pixel=lambda x, y: 64 if (x + y) % 2 else 192))
        self._request_comparison()
        for image in self.manifest["images"]:
            for key in ("exposure_ms", "gain", "processing"):
                del image[key]

        report = self._run()
        self.assertEqual(report["comparison"]["status"], "proxy_ranked")
        self.assertIn(
            report["comparison"]["higher_focus_proxy_image_id"],
            {"synthetic", "second"},
        )
        self.assertTrue(report["human_review_required"])

    def test_deterministic_noise_does_not_remove_review_gate(self) -> None:
        # Noise can make a Laplacian metric larger without improving an image's
        # scientific content. Even a ranked proxy must retain its scope limits.
        moderate_texture = _gray_png(pixel=lambda x, y: 108 if (x + y) % 2 else 148)
        (self.root / "synthetic.png").write_bytes(moderate_texture)
        self.manifest["images"][0]["sha256"] = hashlib.sha256(moderate_texture).hexdigest()
        self.manifest["images"][0]["bytes"] = len(moderate_texture)
        noise = _gray_png(pixel=lambda x, y: ((x * 73 + y * 151) ^ (x * y * 19)) & 255)
        self._add_image("noise", noise)
        self._request_comparison()

        report = self._run()
        self.assertEqual(report["comparison"]["status"], "proxy_ranked")
        by_id = {image["id"]: image for image in report["images"]}
        self.assertGreater(
            by_id["noise"]["metrics"]["focus_proxy"],
            by_id["synthetic"]["metrics"]["focus_proxy"],
        )
        self.assertEqual(report["interpretation_scope"], "observable_image_quality")
        self.assertTrue(report["human_review_required"])
        self.assertEqual(report["atlas"]["status"], "not_configured")
        self.assertEqual(report["biological_assessment"]["status"], "abstain")
        self.assertEqual(report["human_decision"]["status"], "pending")


if __name__ == "__main__":
    unittest.main()
