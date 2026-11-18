"""Synthetic-only, zero-cloud HistoForge training intake admission tests.

No real slide, HistoForge weights, PHI, Modal job, training or network access.
"""

from __future__ import annotations

import copy
import json
import stat
import subprocess
import sys
import tempfile
import unittest
import uuid
import zipfile
from pathlib import Path

import rfc8785

from microscopy_pilot.histoforge_intake import (
    INTAKE_SCHEMA,
    UPSTREAM_BACKEND_SHA,
    UPSTREAM_EXTENSION_SHA,
    UPSTREAM_MODELS_SHA,
    IntakeHold,
    inspect_intake,
    sha_bytes,
)

MODEL_REPO = "histoforge/starter-dinov3-cellpose"
SHA40 = "c" * 40
BASE_BYTES = {
    "artifacts/0/origin.json": b'{"synthetic":true}',
    "artifacts/0/source.png": b"FAKE_IMAGE_ZERO_NO_PIXEL_DATA",
    "artifacts/1/origin.json": b'{"synthetic":true}',
    "artifacts/1/source.png": b"FAKE_IMAGE_ONE_NO_PIXEL_DATA",
    "rois/0/objects.geojson": b'{"type":"FeatureCollection","features":[]}',
    "rois/0/roi.geojson": b'{"type":"FeatureCollection","features":[]}',
    "rois/1/objects.geojson": b'{"type":"FeatureCollection","features":[]}',
    "rois/1/roi.geojson": b'{"type":"FeatureCollection","features":[]}',
}


def sample_manifest():
    files = [
        {"path": name, "size_bytes": len(blob), "sha256": sha_bytes(blob)}
        for name, blob in sorted(BASE_BYTES.items())
    ]
    manifest = {
        "schema_version": 2,
        "task": {
            "capability_id": "synthetic-do-not-run",
            "input_kind": "field",
            "model_output": {"type": "instance_segmentation", "classes": ["Object"]},
            "supports_negative_rois": False,
            "model_microns_per_pixel": 0.5,
            "annotation_class_map": {"Object": "Object"},
        },
        "project_groups": ["fake_specimen_a", "fake_specimen_b"],
        "images": [
            {"project_entry_id_digest": "a" * 64, "project_group_index": 0},
            {"project_entry_id_digest": "b" * 64, "project_group_index": 1},
        ],
        "artifacts": [
            {"image_index": 0, "representation": "png"},
            {"image_index": 1, "representation": "png"},
        ],
        "rois": [
            {
                "source_roi_uuid": str(uuid.UUID(int=i + 1)),
                "image_index": i,
                "source_envelope": {
                    "artifact_index": i,
                    "series_index": 0,
                    "resolution_level": 0,
                    "pixel_rect": {"x": 0, "y": 0, "width": 64, "height": 64},
                },
            }
            for i in (0, 1)
        ],
        "files": files,
    }
    manifest["dataset_digest"] = sha_bytes(rfc8785.dumps(manifest))
    return manifest


def sample_split(manifest):
    return {
        "schema_version": 1,
        "dataset_digest": manifest["dataset_digest"],
        "split_plan": {
            "type": "split",
            "unit": "project_group",
            "assignment_source": "custom",
            "units": [
                {"unit_index": 0, "partition": "training"},
                {"unit_index": 1, "partition": "validation"},
            ],
        },
    }


def sample_request():
    return {
        "schema": INTAKE_SCHEMA,
        "upstream": {
            "backend_commit": UPSTREAM_BACKEND_SHA,
            "models_commit": UPSTREAM_MODELS_SHA,
            "extension_commit": UPSTREAM_EXTENSION_SHA,
        },
        "archive": {"path": "dataset.zip", "size_bytes": 1, "sha256": "0" * 64},
        "split_sidecar": {"path": "split.json", "size_bytes": 1, "sha256": "0" * 64},
        "module": {
            "repo": MODEL_REPO,
            "revision": SHA40,  # Synthetic only, not a verified Hub model revision.
            "model_id": "objects",
            "weight_terms": "UNVERIFIED",
        },
        "permissions": {
            "local_training_allowed": False,
            "cloud_upload_allowed": False,
            "specimen_identity_reviewed": False,
            "publication_allowed": False,
        },
    }


class SyntheticFixture(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def fixture(
        self,
        *,
        manifest=None,
        split=None,
        members=None,
        archive_override=None,
        request_edit=None,
        duplicate_member=None,
    ):
        manifest = copy.deepcopy(
            manifest if manifest is not None else sample_manifest()
        )
        split = copy.deepcopy(split if split is not None else sample_split(manifest))
        members = dict(BASE_BYTES if members is None else members)
        if archive_override is None:
            archive_override = rfc8785.dumps(manifest)
        zip_path = self.root / "dataset.zip"
        with zipfile.ZipFile(
            zip_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True
        ) as package:
            package.writestr("dataset.json", archive_override)
            for path, content in members.items():
                package.writestr(path, content)
            if duplicate_member:
                package.writestr(duplicate_member, members[duplicate_member])
        side_path = self.root / "split.json"
        side_path.write_bytes(rfc8785.dumps(split))
        req = sample_request()
        for key, path in (("archive", zip_path), ("split_sidecar", side_path)):
            raw = path.read_bytes()
            req[key]["size_bytes"] = len(raw)
            req[key]["sha256"] = sha_bytes(raw)
        if request_edit:
            request_edit(req)
        input_path = self.root / "intake.json"
        input_path.write_bytes(rfc8785.dumps(req))
        return input_path

    def assert_hold(self, file, code):
        with self.assertRaises(IntakeHold) as got:
            inspect_intake(file)
        self.assertEqual(str(got.exception), code)


class HistoForgeIntakeTests(SyntheticFixture):
    def test_exact_synthetic_bundle_becomes_nonexecuting_evidence(self):
        result = inspect_intake(self.fixture())
        self.assertEqual(result["status"], "SOURCE_BYTES_AND_SPECIMEN_SPLIT_CHECKED")
        self.assertFalse(result["training_execution_authorized"])
        self.assertFalse(result["upload_authorized"])
        self.assertFalse(result["publication_authorized"])
        self.assertFalse(result["model_inference_qualified"])
        self.assertFalse(result["clinical_diagnostic_qualified"])
        self.assertFalse(result["permission_declarations_are_independent_proof"])
        self.assertEqual(result["data"]["file_count"], 8)
        self.assertEqual(result["data"]["image_count"], 2)
        self.assertEqual(result["data"]["roi_count"], 2)
        self.assertEqual(
            result["data"]["split_counts"]["image_count"],
            {"training": 1, "validation": 1},
        )
        self.assertEqual(
            result["data"]["split_counts"]["roi_count"],
            {"training": 1, "validation": 1},
        )
        self.assertEqual(len(result["next_gates"]), 8)
        body = json.dumps(result)
        for private in (
            "fake_specimen_a",
            "fake_specimen_b",
            "FAKE_IMAGE",
            "c" * 40,
            "Object",
        ):
            self.assertNotIn(private, body)
        self.assertEqual(result, inspect_intake(self.root / "intake.json"))

    def test_self_declared_permissions_cannot_authorize_compute(self):
        def make_true(req):
            req["permissions"] = {name: True for name in req["permissions"]}

        state = inspect_intake(self.fixture(request_edit=make_true))
        self.assertFalse(state["training_execution_authorized"])
        self.assertFalse(state["publication_authorized"])
        self.assertFalse(state["source_provenance_signed"])

    def test_upstream_source_revisions_are_exact_and_review_gated(self):
        def changed(req):
            req["upstream"]["backend_commit"] = "b" * 40

        self.assert_hold(
            self.fixture(request_edit=changed),
            "HISTOFORGE_SOURCE_DRIFT_REVIEW_REQUIRED",
        )

    def test_noncanonical_json_is_not_histo_digest_evidence(self):
        m = sample_manifest()
        raw = json.dumps(m, indent=2).encode("utf-8")
        self.assert_hold(
            self.fixture(archive_override=raw), "NONCANONICAL_HISTOFORGE_DATASET_JSON"
        )

    def test_wrong_dataset_digest_and_synthetic_tampered_task_refused(self):
        m = sample_manifest()
        m["task"]["model_microns_per_pixel"] = 0.75
        self.assert_hold(self.fixture(manifest=m), "DATASET_JCS_DIGEST_MISMATCH")

    def test_pixel_scale_nan_and_zero_refused(self):
        for invalid in (0, -0.5, 1e10, True):
            m = sample_manifest()
            m["task"]["model_microns_per_pixel"] = invalid
            m["dataset_digest"] = sha_bytes(
                rfc8785.dumps({k: v for k, v in m.items() if k != "dataset_digest"})
            )
            with self.subTest(value=invalid):
                self.assert_hold(self.fixture(manifest=m), "MICRONS_PER_PIXEL_INVALID")

    def test_roi_level_can_not_point_at_a_different_image(self):
        m = sample_manifest()
        m["rois"][0]["source_envelope"]["artifact_index"] = 1
        m["dataset_digest"] = sha_bytes(
            rfc8785.dumps({k: v for k, v in m.items() if k != "dataset_digest"})
        )
        self.assert_hold(self.fixture(manifest=m), "CROSS_IMAGE_ARTIFACT_REFERENCE")

    def test_duplicate_zip_member_rejected(self):
        self.assert_hold(
            self.fixture(duplicate_member="rois/0/roi.geojson"),
            "ZIP_DUPLICATE_OR_MISSING_MANIFEST",
        )

    def test_zip_member_hash_mismatch_refused(self):
        altered = dict(BASE_BYTES)
        old = altered["rois/0/roi.geojson"]
        altered["rois/0/roi.geojson"] = b"X" * len(old)
        self.assert_hold(self.fixture(members=altered), "ZIP_ENTRY_SHA256_MISMATCH")

    def test_zip_path_traversal_refused(self):
        altered = dict(BASE_BYTES)
        altered["../malicious.txt"] = b"not accepted"
        self.assert_hold(self.fixture(members=altered), "UNSAFE_RELATIVE_PATH")

    def test_zip_symlink_refused(self):
        m = sample_manifest()
        raw = rfc8785.dumps(m)
        path = self.root / "dataset.zip"
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("dataset.json", raw)
            for name, contents in BASE_BYTES.items():
                if name == "rois/0/roi.geojson":
                    info = zipfile.ZipInfo(name)
                    info.create_system = 3
                    info.external_attr = (stat.S_IFLNK | 0o777) << 16
                    z.writestr(info, b"target")
                else:
                    z.writestr(name, contents)
        self.fixture()
        req = json.loads((self.root / "intake.json").read_text())
        # Overwrite test archive after fixture creation; preserve expected metadata.
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("dataset.json", raw)
            for name, contents in BASE_BYTES.items():
                info = zipfile.ZipInfo(name)
                if name == "rois/0/roi.geojson":
                    info.create_system = 3
                    info.external_attr = (stat.S_IFLNK | 0o777) << 16
                z.writestr(info, contents)
        blob = path.read_bytes()
        req["archive"]["size_bytes"] = len(blob)
        req["archive"]["sha256"] = sha_bytes(blob)
        (self.root / "intake.json").write_bytes(rfc8785.dumps(req))
        self.assert_hold(self.root / "intake.json", "ZIP_SYMLINK_FORBIDDEN")

    def test_split_roi_level_leakage_is_refused(self):
        m = sample_manifest()
        side = sample_split(m)
        side["split_plan"]["unit"] = "roi"
        self.assert_hold(
            self.fixture(manifest=m, split=side), "SPECIMEN_LEVEL_SPLIT_REQUIRED"
        )

    def test_train_only_split_is_refused(self):
        m = sample_manifest()
        side = sample_split(m)
        side["split_plan"]["units"][1]["partition"] = "training"
        self.assert_hold(
            self.fixture(manifest=m, split=side), "TRAIN_AND_VALIDATION_REQUIRED"
        )

    def test_split_must_bind_exact_dataset_digest(self):
        m = sample_manifest()
        side = sample_split(m)
        side["dataset_digest"] = "0" * 64
        self.assert_hold(
            self.fixture(manifest=m, split=side), "SPLIT_DATASET_DIGEST_MISMATCH"
        )

    def test_missing_group_and_image_identity_stops(self):
        m = sample_manifest()
        del m["images"][0]["project_group_index"]
        m["dataset_digest"] = sha_bytes(
            rfc8785.dumps({k: v for k, v in m.items() if k != "dataset_digest"})
        )
        self.assert_hold(self.fixture(manifest=m), "IMAGE_GROUP_BINDING_REQUIRED")

    def test_module_requires_exact_ref_and_unverified_license(self):
        def missing_pin(req):
            req["module"]["revision"] = "main"

        self.assert_hold(self.fixture(request_edit=missing_pin), "UNPINNED_MODULE")

        def arbitrary_license(req):
            req["module"]["weight_terms"] = "commercial-approved"

        self.assert_hold(
            self.fixture(request_edit=arbitrary_license),
            "MODEL_LICENSE_NOT_INDEPENDENTLY_VERIFIED",
        )

    def test_foreign_project_or_path_escape_cannot_enter_source(self):
        def fake_repo(req):
            req["module"]["repo"] = "unsafe/private-dataset"

        self.assert_hold(
            self.fixture(request_edit=fake_repo), "MODULE_REPO_NOT_ADMITTED"
        )

        def escape(req):
            req["archive"]["path"] = "../out.zip"

        self.assert_hold(self.fixture(request_edit=escape), "UNSAFE_RELATIVE_PATH")

    def test_boolean_counts_and_unknown_request_fields_refused(self):
        def invalid_bytes(req):
            req["archive"]["size_bytes"] = True

        self.assert_hold(self.fixture(request_edit=invalid_bytes), "INPUT_SIZE_INVALID")

        def extra(req):
            req["private_slide_path"] = "C:/patient-123/file"

        self.assert_hold(self.fixture(request_edit=extra), "INTAKE_REQUEST_SCHEMA")

    def test_archive_bytes_do_not_match_declared_source(self):
        request = self.fixture()
        payload = json.loads(request.read_bytes())
        payload["archive"]["sha256"] = "f" * 64
        request.write_bytes(rfc8785.dumps(payload))
        self.assert_hold(request, "INPUT_SHA256_MISMATCH")

    def test_split_sidecar_must_be_jcs_canonical(self):
        request = self.fixture()
        payload = json.loads(request.read_bytes())
        sidecar = self.root / "split.json"
        pretty = json.dumps(json.loads(sidecar.read_bytes()), indent=2).encode()
        sidecar.write_bytes(pretty)
        payload["split_sidecar"]["size_bytes"] = len(pretty)
        payload["split_sidecar"]["sha256"] = sha_bytes(pretty)
        request.write_bytes(rfc8785.dumps(payload))
        self.assert_hold(request, "NONCANONICAL_HISTOFORGE_SPLIT")

    def test_duplicate_json_keys_refused(self):
        request = self.fixture()
        request.write_bytes(b'{"schema":"first","schema":"second"}')
        self.assert_hold(request, "DUPLICATE_JSON_KEY")

    def test_archive_member_closure_must_match_the_manifest(self):
        m = sample_manifest()
        altered = dict(BASE_BYTES)
        altered["artifacts/7/secret.txt"] = b"not declared"
        self.assert_hold(
            self.fixture(manifest=m, members=altered),
            "ZIP_MANIFEST_CLOSURE_MISMATCH",
        )

    def test_local_symlink_does_not_gain_access(self):
        request = self.fixture()
        original = Path.is_symlink

        def mark_archive_as_symlink(path):
            if path.name == "dataset.zip":
                return True
            return original(path)

        from unittest.mock import patch

        with patch.object(Path, "is_symlink", mark_archive_as_symlink):
            self.assert_hold(request, "SYMLINK_INPUT_FORBIDDEN")

    def test_existing_package_cli_wires_histoforge_admit_without_dispatch(self):
        request = self.fixture()
        out = self.root / "from-existing-cli.json"
        command = [
            sys.executable,
            "-m",
            "microscopy_pilot",
            "histoforge-admit",
            "--request",
            str(request),
            "--out",
            str(out),
        ]
        process = subprocess.run(
            command, capture_output=True, text=True, timeout=15, check=False
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        result = json.loads(out.read_bytes())
        self.assertEqual(result["claim"], "OFFLINE_TRAINING_INTAKE_ONLY")
        self.assertEqual(result["status"], "SOURCE_BYTES_AND_SPECIMEN_SPLIT_CHECKED")
        self.assertFalse(result["training_execution_authorized"])
        self.assertFalse(result["upload_authorized"])

    def test_histo_source_requires_jcs_sorted_project_groups(self):
        manifest = sample_manifest()
        manifest["project_groups"].reverse()
        manifest["dataset_digest"] = sha_bytes(
            rfc8785.dumps({k: v for k, v in manifest.items() if k != "dataset_digest"})
        )
        self.assert_hold(
            self.fixture(manifest=manifest),
            "PROJECT_GROUP_ORDER_NOT_CANONICAL",
        )

    def test_derived_artifact_must_pair_with_exact_roi(self):
        from microscopy_pilot.histoforge_intake import manifest_checks

        manifest = sample_manifest()
        manifest["artifacts"] = [
            {"image_index": 0, "representation": "png"},
            {"image_index": 0, "representation": "png"},
            {"image_index": 1, "representation": "png"},
        ]
        first = copy.deepcopy(manifest["rois"][0])
        duplicate = copy.deepcopy(first)
        duplicate["source_roi_uuid"] = str(uuid.UUID(int=3))
        # This second ROI for image zero incorrectly reuses artifact 0;
        # count-only validation would miss the mismatch.
        duplicate["source_envelope"]["artifact_index"] = 0
        second = copy.deepcopy(manifest["rois"][1])
        second["source_envelope"]["artifact_index"] = 2
        manifest["rois"] = [first, duplicate, second]
        filenames = [
            f"artifacts/{idx}/{leaf}"
            for idx in range(3)
            for leaf in ("origin.json", "source.png")
        ] + [
            f"rois/{idx}/{leaf}"
            for idx in range(3)
            for leaf in ("objects.geojson", "roi.geojson")
        ]
        manifest["files"] = [
            {"path": name, "size_bytes": 1, "sha256": "a" * 64}
            for name in sorted(filenames)
        ]
        manifest["dataset_digest"] = sha_bytes(
            rfc8785.dumps({k: v for k, v in manifest.items() if k != "dataset_digest"})
        )
        with self.assertRaisesRegex(IntakeHold, "DERIVED_ARTIFACT_PAIR_MISMATCH"):
            manifest_checks(manifest)

    def test_zip_preflight_never_calls_unbounded_testzip(self):
        from unittest.mock import patch

        request = self.fixture()
        with patch.object(
            zipfile.ZipFile,
            "testzip",
            side_effect=AssertionError("testzip bypassed safety budget"),
        ):
            result = inspect_intake(request)
        self.assertEqual(result["data"]["file_count"], 8)
        self.assertFalse(result["training_execution_authorized"])

    def test_cli_exclusive_create_and_secret_free_output(self):
        request = self.fixture()
        out = self.root / "admission.json"
        command = [
            sys.executable,
            "-m",
            "microscopy_pilot.histoforge_intake",
            "--request",
            str(request),
            "--out",
            str(out),
        ]
        first = subprocess.run(
            command, capture_output=True, text=True, timeout=15, check=False
        )
        self.assertEqual(first.returncode, 0, first.stderr)
        evidence = json.loads(out.read_text())
        self.assertFalse(evidence["training_execution_authorized"])
        self.assertNotIn("fake_specimen_a", out.read_text())
        before = out.read_bytes()
        again = subprocess.run(
            command, capture_output=True, text=True, timeout=15, check=False
        )
        self.assertEqual(again.returncode, 2)
        self.assertEqual(before, out.read_bytes())
        self.assertEqual(
            json.loads(again.stdout)["reason_code"], "FILE_OR_OUTPUT_UNAVAILABLE"
        )


if __name__ == "__main__":
    unittest.main()
