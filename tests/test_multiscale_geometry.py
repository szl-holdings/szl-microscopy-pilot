"""Synthetic, dependency-free multiscale spatial integrity tests."""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from microscopy_pilot.multiscale_geometry import GeometryError, audit_multiscale_geometry, read_manifest


def fixture(root: Path) -> dict:
    levels = []
    for idx, (shape, scale) in enumerate((([4, 6, 8], [1.5, 0.8, 0.3]),
                                         ([2, 3, 4], [3, 1.6, 0.6]))):
        body = bytes((i + idx * 101) % 256 for i in range(shape[0] * shape[1] * shape[2]))
        name = f"level_{idx}.raw"
        (root / name).write_bytes(body)
        levels.append({"path": name, "shape": shape, "scale_um": scale,
                       "translation_um": [0, 0, 0], "sha256": hashlib.sha256(body).hexdigest()})
    return {"schema": "szl.microscopy.synthetic-multiscale/v1",
            "identity": {"specimen_id": "synthetic-only", "channel": "nuclear", "source_revision": "synthetic"},
            "axes": [{"name": d, "type": "space", "unit": "micrometer"} for d in ("z", "y", "x")],
            "levels": levels}


class MultiscaleGeometryTests(unittest.TestCase):
    def test_round_trip_synthetic_files_and_deterministic_physical_bounds(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest = fixture(root)
            (root / "manifest.json").write_text(json.dumps(manifest))
            a = audit_multiscale_geometry(read_manifest(root / "manifest.json"), root)
            b = audit_multiscale_geometry(read_manifest(root / "manifest.json"), root)
            self.assertEqual(a, b)
            self.assertEqual([[0, 6], [0, 4.8], [0, 2.4]], a["world_bounds_um"])
            self.assertEqual(216, a["total_bytes"])
            self.assertFalse(a["ngff_conformance_verified"])
            self.assertFalse(a["holographic_runtime_verified"])
            self.assertEqual("ABSTAIN", a["biological_interpretation"])

    def test_axis_order_swap_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            x = fixture(root)
            x["axes"][0], x["axes"][2] = x["axes"][2], x["axes"][0]
            with self.assertRaisesRegex(GeometryError, "AXIS_ORDER"):
                audit_multiscale_geometry(x, root)

    def test_downsample_scale_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            x = fixture(root)
            x["levels"][1]["scale_um"][2] = 0.9
            with self.assertRaisesRegex(GeometryError, "INCONSISTENT_DOWNSAMPLE_SCALE"):
                audit_multiscale_geometry(x, root)

    def test_translation_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            x = fixture(root)
            x["levels"][1]["translation_um"][1] = 0.5
            with self.assertRaisesRegex(GeometryError, "WORLD_BOUNDS_MISMATCH"):
                audit_multiscale_geometry(x, root)

    def test_byte_corruption_rejected_without_loading_models(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            x = fixture(root)
            (root / x["levels"][1]["path"]).write_bytes(b"x" * 24)
            with self.assertRaisesRegex(GeometryError, "LEVEL_DATA_MISMATCH"):
                audit_multiscale_geometry(x, root)

    def test_missing_metadata_and_bad_downsample_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            x = fixture(root)
            del x["identity"]["channel"]
            with self.assertRaises(GeometryError):
                audit_multiscale_geometry(x, root)
            x = fixture(root)
            x["levels"][1]["shape"] = [2, 4, 4]
            with self.assertRaises(GeometryError):
                audit_multiscale_geometry(x, root)

    def test_duplicate_json_keys_and_nan_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            p = root / "manifest.json"
            for payload in (b'{"axes": [], "axes": []}', b'{"levels": [NaN]}', b'{"levels": [1e999]}'):
                p.write_bytes(payload)
                with self.subTest(payload=payload), self.assertRaises(GeometryError):
                    read_manifest(p)

    def test_cli_denies_bad_manifest_without_writing_anything(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            x = fixture(root)
            p = root / "manifest.json"
            p.write_text(json.dumps(x))
            cmd = [sys.executable, "-m", "microscopy_pilot.multiscale_geometry", "--manifest", str(p)]
            ok = subprocess.run(cmd, capture_output=True, text=True, check=False)
            self.assertEqual(0, ok.returncode, ok.stderr)
            self.assertEqual("SYNTHETIC_COORDINATES_AND_BYTES_PASSED", json.loads(ok.stdout)["status"])
            p.write_text("{}")
            no = subprocess.run(cmd, capture_output=True, text=True, check=False)
            self.assertEqual(2, no.returncode, no.stderr)
            self.assertEqual("DENY", json.loads(no.stdout)["status"])


if __name__ == "__main__":
    unittest.main()
