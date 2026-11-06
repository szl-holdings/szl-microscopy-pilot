"""The README's quickstart commands, run as subprocesses against the native example.

Each command runs offline with the current interpreter from the repository
root, exactly as documented; no network is used.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "examples/bbbc006v1/manifest.json"


def _cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "microscopy_pilot", *args],
        cwd=ROOT, capture_output=True, text=True, timeout=300, check=False,
    )


class CommandLineTests(unittest.TestCase):
    def test_verify_checks_the_native_pair_without_writing_a_report(self) -> None:
        result = _cli("verify", "--manifest", MANIFEST)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            json.loads(result.stdout),
            {"status": "verified", "images": 2, "comparison": "proxy_ranked", "report": None},
        )

    def test_run_writes_a_report_that_keeps_the_review_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "report.json"
            result = _cli("run", "--manifest", MANIFEST, "--out", str(out))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["report"], str(out))
            report = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(report["comparison"]["higher_focus_proxy_image_id"], "a02_s1_w1_z16")
        self.assertTrue(report["human_review_required"])
        self.assertEqual(report["human_decision"]["status"], "pending")
        self.assertEqual(report["biological_assessment"]["status"], "abstain")

    def test_a_failed_gate_exits_2_with_its_reason_and_no_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.json"
            manifest.write_text(json.dumps({"claim": "disease_diagnosis"}), encoding="utf-8")
            out = Path(tmp) / "report.json"
            result = _cli("run", "--manifest", str(manifest), "--out", str(out))
            self.assertFalse(out.exists())
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr, "pilot: only observable_quality_only claims are supported\n")

    def test_a_missing_subcommand_is_a_usage_error(self) -> None:
        result = _cli()
        self.assertEqual(result.returncode, 2)
        self.assertIn("usage:", result.stderr)


if __name__ == "__main__":
    unittest.main()
