"""Offline exact-decoder and retained-example compatibility regressions."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import PIL
from PIL import Image  # Initialize the native decoder before version-mismatch mocking.

from microscopy_pilot.core import PilotError, _measure_image, run


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "examples/bbbc006v1/manifest.json"


class DecoderCompatibilityTests(unittest.TestCase):
    def test_pinned_decoder_preserves_native_example_measurements(self):
        self.assertEqual(PIL.__version__, "12.3.0")
        historical = json.loads((ROOT / "examples/bbbc006v1/report.json").read_text())
        expected = copy.deepcopy(historical)
        expected["method"]["parameters"]["decoder"] = "Pillow 12.3.0"
        self.assertEqual(run(MANIFEST), expected)

    def test_supported_decode_paths_emit_no_deprecation_warnings(self):
        # Image.getdata() is deprecated (removed in Pillow 14); every supported
        # mode must decode without touching a deprecated Pillow API.
        with tempfile.TemporaryDirectory() as tmp:
            gray = Path(tmp) / "gray.png"
            rgb = Path(tmp) / "rgb.png"
            Image.new("L", (5, 4), 17).save(gray)
            Image.new("RGB", (5, 4), (40, 90, 200)).save(rgb)
            with warnings.catch_warnings():
                warnings.simplefilter("error", DeprecationWarning)
                run(MANIFEST)  # native 16-bit TIFF example
                self.assertEqual(_measure_image(gray)["bit_depth"], 8)
                self.assertEqual(_measure_image(rgb)["bit_depth"], 8)

    def test_rgb_luma_path_matches_grayscale_path_for_neutral_pixels(self):
        # Rec. 709 luma weights sum to 1, so a neutral RGB image must measure
        # exactly like the same values decoded through the 8-bit grayscale path.
        values = [(x * x * 37 + y * 11 + (x ^ y) * 5) % 256 for y in range(6) for x in range(7)]
        with tempfile.TemporaryDirectory() as tmp:
            gray = Image.new("L", (7, 6))
            gray.putdata(values)
            gray.save(Path(tmp) / "gray.png")
            rgb = Image.new("RGB", (7, 6))
            rgb.putdata([(v, v, v) for v in values])
            rgb.save(Path(tmp) / "rgb.png")
            gray_metrics = _measure_image(Path(tmp) / "gray.png")
            rgb_metrics = _measure_image(Path(tmp) / "rgb.png")
        self.assertGreater(gray_metrics["laplacian_variance"], 0)
        self.assertEqual(rgb_metrics, gray_metrics)

    def test_an_unqualified_decoder_version_still_fails_closed(self):
        with patch.object(PIL, "__version__", "11.3.0"):
            with self.assertRaisesRegex(PilotError, "requires Pillow==12.3.0"):
                run(MANIFEST)


if __name__ == "__main__":
    unittest.main()
