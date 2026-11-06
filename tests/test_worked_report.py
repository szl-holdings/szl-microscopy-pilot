"""Offline regressions for the documented static worked-report build.

The README rebuilds docs/worked_report.html from the retained BBBC006v1 report
and TIFFs. The retained report keeps its original decoder provenance, so a
fresh run under the pinned decoder differs in method.parameters.decoder and in
nothing else. These tests pin that rule, the committed page, and the --check
mode. No network is used and no measurement is changed.
"""

from __future__ import annotations

import base64
import contextlib
import copy
import io
import json
from pathlib import Path
import re
import shutil
import tempfile
import unittest
from unittest.mock import patch
import warnings

from PIL import Image

from microscopy_pilot.core import PilotError, run
from tools import build_worked_report as worked


# docs/WORKED_EXAMPLE.md states this pooled 99.9th-percentile display white.
DOCUMENTED_DISPLAY_WHITE = 1110


def _set(document: dict, path: tuple, value) -> None:
    for key in path[:-1]:
        document = document[key]
    document[path[-1]] = value


class WorkedReportBuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            cls.page, cls.white = worked.render()
        cls.fresh = run(worked.MANIFEST)
        cls.recorded = json.loads(worked.REPORT.read_text(encoding="utf-8"))

    def _main(self, *argv: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with patch.object(worked, "render", return_value=(self.page, self.white)), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = worked.main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_documented_rebuild_reproduces_the_committed_page(self) -> None:
        committed = worked.OUTPUT.read_text(encoding="utf-8")
        self.assertEqual(worked.page_fingerprint(committed), worked.page_fingerprint(self.page))
        self.assertEqual(self.white, DOCUMENTED_DISPLAY_WHITE)
        self.assertEqual(len(worked.PNG_DATA_URI.findall(self.page)), 2)

    def test_only_decoder_provenance_may_differ_from_a_fresh_run(self) -> None:
        worked.check_recorded_report(self.fresh, self.recorded)
        tampered_fields = (
            (("images", 0, "metrics", "focus_proxy"), 0.5),
            (("images", 1, "sha256"), "0" * 64),
            (("method", "parameters", "max_pixels"), 1),
            (("comparison", "higher_focus_proxy_image_id"), "a02_s1_w1_z00"),
            (("human_decision", "status"), "keep"),
        )
        for path, value in tampered_fields:
            tampered = copy.deepcopy(self.recorded)
            _set(tampered, path, value)
            with self.subTest(path=path), \
                    self.assertRaisesRegex(ValueError, "differs from a fresh verified run"):
                worked.check_recorded_report(self.fresh, tampered)
        for value in (None, ""):
            missing = copy.deepcopy(self.recorded)
            if value is None:
                del missing["method"]["parameters"]["decoder"]
            else:
                missing["method"]["parameters"]["decoder"] = value
            with self.subTest(decoder=value), self.assertRaisesRegex(ValueError, "decoder provenance"):
                worked.check_recorded_report(self.fresh, missing)

    def test_damaged_source_tiff_cannot_be_previewed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            example = Path(tmp) / "bbbc006v1"
            shutil.copytree(worked.MANIFEST.parent, example)
            manifest = json.loads((example / "manifest.json").read_text(encoding="utf-8"))
            tiff = example / manifest["images"][0]["path"]
            data = bytearray(tiff.read_bytes())
            data[-1] ^= 0xFF  # same length, different bytes
            tiff.write_bytes(bytes(data))
            with self.assertRaisesRegex(PilotError, "SHA-256 mismatch"):
                worked.render(example / "manifest.json", example / "report.json")

    def test_check_accepts_the_committed_page_and_writes_nothing(self) -> None:
        before = worked.OUTPUT.read_bytes()
        code, out, _ = self._main("--check")
        self.assertEqual(code, 0)
        self.assertIn("up to date", out)
        self.assertEqual(worked.OUTPUT.read_bytes(), before)

    def test_check_accepts_reencoded_previews_with_identical_pixels(self) -> None:
        first = worked.PNG_DATA_URI.search(self.page)
        with Image.open(io.BytesIO(base64.b64decode(first[1]))) as image:
            stream = io.BytesIO()
            image.save(stream, format="PNG", compress_level=1)
        reencoded = base64.b64encode(stream.getvalue()).decode("ascii")
        self.assertNotEqual(reencoded, first[1])
        with tempfile.TemporaryDirectory() as tmp:
            page = Path(tmp) / "worked_report.html"
            page.write_text(self.page.replace(first[1], reencoded, 1), encoding="utf-8")
            code, out, _ = self._main("--check", "--output", str(page))
        self.assertEqual(code, 0)
        self.assertIn("decoded pixels", out)

    def test_check_rejects_changed_text_or_preview_pixels(self) -> None:
        first = worked.PNG_DATA_URI.search(self.page)
        with Image.open(io.BytesIO(base64.b64decode(first[1]))) as image:
            size = image.size
        stream = io.BytesIO()
        Image.new("L", size, 0).save(stream, format="PNG")
        black = base64.b64encode(stream.getvalue()).decode("ascii")
        variants = {
            "text": self.page.replace("Observed result", "Observed results", 1),
            "pixels": self.page.replace(first[1], black, 1),
            "corrupt": self.page.replace(first[1], "AAAA", 1),
        }
        for name, content in variants.items():
            with self.subTest(variant=name), tempfile.TemporaryDirectory() as tmp:
                page = Path(tmp) / "worked_report.html"
                page.write_text(content, encoding="utf-8")
                code, out, err = self._main("--check", "--output", str(page))
                self.assertEqual(code, 1)
                self.assertEqual(out, "")
                self.assertTrue(re.search("stale|cannot compare", err))

    def test_output_option_writes_the_page_elsewhere(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "nested" / "page.html"
            code, out, _ = self._main("--output", str(target))
            self.assertEqual(code, 0)
            self.assertEqual(target.read_text(encoding="utf-8"), self.page)
        self.assertIn(f"shared display white={DOCUMENTED_DISPLAY_WHITE}", out)


if __name__ == "__main__":
    unittest.main()
