"""No-cloud, no-weight verification of literature->experiment evidence contracts."""
from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from microscopy_pilot.pathology_frontier import Hold, study_plan
from microscopy_pilot.pathology_metrics import (
    MetricHold, instance_scores, paired_specimen_bootstrap,
)

REGISTRY = Path(__file__).resolve().parents[1] / "research" / "pathology_frontier_v1.json"


def doc():
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def plan(value):
    return study_plan(json.dumps(value, sort_keys=True).encode("utf-8"))


class LiteratureAndExperimentTests(unittest.TestCase):
    def assert_refused(self, value, reason):
        with self.assertRaises(Hold) as caught:
            plan(value)
        self.assertEqual(str(caught.exception), reason)

    def test_exact_source_registry_is_advisory_only(self):
        result = study_plan(REGISTRY.read_bytes())
        self.assertEqual(result["literature_count"], 15)
        self.assertEqual(result["candidate_count"], 9)
        self.assertEqual(result["status"], "SOURCE_PINNED_ADVISORY_ONLY")
        self.assertEqual(len(result["registry_jcs_sha256"]), 64)
        self.assertIsNone(result["benchmark_results"])
        for name in ("training_authorized", "model_publication_authorized",
                     "model_weights_downloaded", "signatures_verified",
                     "clinical_use_authorized"):
            self.assertFalse(result[name])
        for candidate in result["experiment_candidates"]:
            self.assertFalse(candidate["action_authorized"])
            self.assertEqual(candidate["training_steps_completed"], 0)
            self.assertEqual(candidate["evaluation_state"], "NOT_EXECUTED")
            for binding in candidate["source_bindings"]:
                self.assertTrue(binding["read_only_url"].endswith(binding["git_commit"]))
                self.assertEqual(len(binding["git_commit"]), 40)
        self.assertEqual(result, study_plan(REGISTRY.read_bytes()))

    def test_no_unpinned_head_or_unknown_weights(self):
        value = doc()
        value["sources"][0]["sha"] = "main"
        self.assert_refused(value, "DUPLICATE_OR_UNPINNED_SOURCE")

    def test_model_and_training_execution_flags_cannot_be_promoted(self):
        value = doc()
        value["arms"][0]["training_authorized"] = True
        self.assert_refused(value, "UNQUALIFIED_MODEL_OR_JOB")
        value = doc()
        value["policy"]["gpu_job_allowed"] = True
        self.assert_refused(value, "EXECUTION_OR_PUBLICATION_CLAIM_FORBIDDEN")

    def test_slide_auc_cannot_be_used_for_instance_nuclei(self):
        value = doc()
        value["arms"][0]["metrics"] = ["AUROC"]
        self.assert_refused(value, "CROSS_TASK_METRICS")

    def test_cross_task_baseline_denied(self):
        value = doc()
        arm = next(a for a in value["arms"] if a["id"] == "celltype_head")
        arm["baseline"] = "hover_offset"
        self.assert_refused(value, "CROSS_TASK_BASELINE")

    def test_exact_three_way_patient_group_split_required(self):
        value = doc()
        value["validation"]["unit"] = "patch"
        self.assert_refused(value, "SCIENTIFIC_SPLIT_OR_SCALE_GATE")
        value = doc()
        value["validation"]["partitions"] = ["training", "validation"]
        self.assert_refused(value, "SCIENTIFIC_SPLIT_OR_SCALE_GATE")

    def test_gated_license_and_budget_are_not_inferred(self):
        value = doc()
        value["policy"]["required_gates"].remove("SIGNED_GPU_CAPACITY_BUDGET")
        self.assert_refused(value, "RIGHTS_AND_EVIDENCE_GATES_MISSING")

    def test_paper_url_never_accepts_untrusted_domains_or_tokens(self):
        for url in ("http://doi.org/paper",
                    "https://doi.org@attacker.example/ref",
                    "https://doi.org/ref?token=private"):
            value = doc()
            value["sources"][0]["paper"] = url
            with self.subTest(url=url):
                self.assert_refused(value, "PAPER_REFERENCE_INVALID")

    def test_json_duplicate_key_and_unknown_top_level_denied(self):
        with self.assertRaisesRegex(Hold, "DUPLICATE_KEY"):
            study_plan(b'{"schema":"1","schema":"2"}')
        value = doc()
        value["credential"] = "must never be accepted"
        self.assert_refused(value, "REGISTRY_SCHEMA")

    def test_cli_exclusive_create_rejects_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "plan.json"
            cmd = [sys.executable, "-m", "microscopy_pilot.pathology_frontier",
                   "--registry", str(REGISTRY), "--out", str(output)]
            first = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            self.assertEqual(first.returncode, 0, first.stderr)
            doc = json.loads(output.read_text())
            self.assertEqual(doc["candidate_count"], 9)
            self.assertFalse(doc["training_authorized"])
            before = output.read_bytes()
            second = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            self.assertEqual(second.returncode, 2)
            self.assertEqual(before, output.read_bytes())


class OriginalBenchmarkMathTests(unittest.TestCase):
    def test_exact_perfect_pq_with_different_instance_ids(self):
        x = instance_scores([[1, 1, 0], [0, 2, 2]],
                            [[9, 9, 0], [0, 8, 8]], mpp_x=0.5, mpp_y=0.25)
        self.assertEqual(x["pq"], 1.0)
        self.assertEqual(x["foreground_dice"], 1.0)
        self.assertEqual(x["true_positives"], 2)
        self.assertEqual(x["gt_area_um2"], 0.5)
        self.assertFalse(x["independent_model_evaluation_proved"])

    def test_exact_half_iou_is_not_a_match(self):
        x = instance_scores([[1, 1, 0]], [[7, 0, 0]],
                            mpp_x=0.5, mpp_y=0.5)
        self.assertEqual(x["pq"], 0.0)
        self.assertEqual(x["false_negatives"], 1)
        self.assertEqual(x["false_positives"], 1)
        self.assertAlmostEqual(x["foreground_dice"], 2/3)

    def test_perfect_foreground_overlap_does_not_mean_correct_instances(self):
        x = instance_scores([[1, 1, 2, 2]], [[9, 9, 9, 9]],
                            mpp_x=0.5, mpp_y=0.5)
        self.assertEqual(x["foreground_dice"], 1.0)
        self.assertEqual(x["pq"], 0.0)

    def test_empty_denominators_are_not_invented_perfection(self):
        x = instance_scores([[0, 0]], [[0, 0]], mpp_x=0.5, mpp_y=0.5)
        self.assertIsNone(x["pq"])
        self.assertIsNone(x["foreground_dice"])

    def test_bad_scale_or_mask_refused(self):
        for bad in (0, float("nan"), True):
            with self.subTest(bad=bad), self.assertRaisesRegex(
                MetricHold, "INVALID_PHYSICAL_SCALE"
            ):
                instance_scores([[1]], [[1]], mpp_x=bad, mpp_y=0.5)
        with self.assertRaisesRegex(MetricHold, "INVALID_INSTANCE_LABEL"):
            instance_scores([[True]], [[1]], mpp_x=0.5, mpp_y=0.5)

    def test_specimen_bootstrap_is_repeatable_and_not_a_superiority_claim(self):
        rows = [{"specimen_hmac_sha256": f"{i+1:064x}",
                 "baseline_pq": 0.25, "candidate_pq": 0.35}
                for i in range(5)]
        a = paired_specimen_bootstrap(rows)
        self.assertEqual(a, paired_specimen_bootstrap(copy.deepcopy(rows)))
        self.assertAlmostEqual(a["mean_delta_pq"], 0.10)
        self.assertFalse(a["superiority_qualified"])
        self.assertFalse(a["actual_patient_distinctness_proved"])
        self.assertEqual(a["paired_specimens"], 5)
        rows[4]["specimen_hmac_sha256"] = rows[0]["specimen_hmac_sha256"]
        with self.assertRaisesRegex(MetricHold, "DUPLICATED_OR_INVALID_SPECIMEN_ID"):
            paired_specimen_bootstrap(rows)


if __name__ == "__main__":
    unittest.main()
