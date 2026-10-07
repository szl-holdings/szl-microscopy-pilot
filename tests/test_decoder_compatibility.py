"""Offline exact-decoder and retained-example compatibility regressions."""

import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import PIL
from PIL import Image  # Initialize the native decoder before version-mismatch mocking.

from microscopy_pilot.core import PilotError, run


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "examples/bbbc006v1/manifest.json"


class DecoderCompatibilityTests(unittest.TestCase):
    def test_pinned_decoder_preserves_native_example_measurements(self):
        self.assertEqual(PIL.__version__, "12.3.0")
        historical = json.loads((ROOT / "examples/bbbc006v1/report.json").read_text())
        expected = copy.deepcopy(historical)
        expected["method"]["parameters"]["decoder"] = "Pillow 12.3.0"
        self.assertEqual(run(MANIFEST), expected)

    def test_an_unqualified_decoder_version_still_fails_closed(self):
        with patch.object(PIL, "__version__", "11.3.0"):
            with self.assertRaisesRegex(PilotError, "requires Pillow==12.3.0"):
                run(MANIFEST)


if __name__ == "__main__":
    unittest.main()
