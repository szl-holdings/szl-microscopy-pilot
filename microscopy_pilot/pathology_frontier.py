"""Fail-closed, offline pathology literature-to-training experiment planner.

Reads only a reviewed source-pinned study registry. Does not import/execute
third-party model code, download weights, submit GPU jobs or claim test results.
The SHA-256 of canonical JSON binds source bytes; it is NOT a signature.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit

import rfc8785

SCHEMA = "szl.pathology-frontier-registry/v1"
OUTPUT_SCHEMA = "szl.pathology-frontier-plan/v1"
SOURCE_KEYS = {"id", "repo", "sha", "paper", "idea", "license_review"}
ARM_KEYS = {"id", "task", "source_ids", "baseline", "metrics", "objective",
            "source_only", "training_authorized", "model_weights_admitted", "benchmarked"}
REGISTRY_KEYS = {"schema", "reviewed_utc", "policy", "validation", "sources", "arms"}
POLICY_KEYS = {"claim", "code_copied", "weights_downloaded", "dataset_downloaded",
               "training_started", "gpu_job_allowed", "cloud_upload_allowed",
               "publication_allowed", "clinical_use_allowed",
               "model_rights_independently_verified", "required_gates"}
VALIDATION_KEYS = {"partitions", "unit", "site_stain_shift_required",
                   "physical_microns_per_pixel_required",
                   "instance_iou_threshold_strictly_above",
                   "bootstrap", "metric_domains", "reported_scores"}
METRICS = {
    "instance": {"PQ", "Dice", "detection_F1"},
    "cell_classification": {"macro_F1", "ece"},
    "semantic": {"Dice", "ece"},
    "slide": {"AUROC", "AUPRC", "ece"},
}
DOMAINS = {"github.com", "histoforge.github.io", "doi.org", "ai.meta.com",
           "proceedings.mlr.press", "www.biorxiv.org"}
SOURCE_RE = re.compile(r"[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+\Z")
SHA_RE = re.compile(r"[a-f0-9]{40}\Z")
ID_RE = re.compile(r"[a-z][a-z0-9_]{2,50}\Z")
MAX_BYTES = 100_000


class Hold(ValueError):
    """Privacy-safe, deterministic refusal code."""


def stop(reason: str) -> None:
    raise Hold(reason)


def closed(value, keys, reason):
    if not isinstance(value, dict) or set(value) != keys:
        stop(reason)
    return value


def strict_json(raw: bytes):
    if not isinstance(raw, bytes) or not 0 < len(raw) <= MAX_BYTES:
        stop("INPUT_LIMIT")
    def pair_object(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                stop("DUPLICATE_KEY")
            obj[key] = value
        return obj
    try:
        return json.loads(raw, object_pairs_hook=pair_object,
                          parse_constant=lambda _: stop("NONFINITE_JSON"))
    except (ValueError, UnicodeError) as exc:
        if isinstance(exc, Hold):
            raise
        stop("INVALID_JSON")


def study_plan(raw: bytes) -> dict:
    doc = closed(strict_json(raw), REGISTRY_KEYS, "REGISTRY_SCHEMA")
    if doc["schema"] != SCHEMA:
        stop("UNSUPPORTED_VERSION")
    if not isinstance(doc["reviewed_utc"], str) or not doc["reviewed_utc"].endswith("Z"):
        stop("REVIEW_TIMESTAMP")
    policy = closed(doc["policy"], POLICY_KEYS, "POLICY_SCHEMA")
    if policy["claim"] != "SOURCE_PINNED_LITERATURE_ADVISORY_ONLY":
        stop("UNVERIFIED_CLAIM")
    for flag in POLICY_KEYS - {"claim", "required_gates"}:
        if policy[flag] is not False:
            stop("EXECUTION_OR_PUBLICATION_CLAIM_FORBIDDEN")
    required = policy["required_gates"]
    if not isinstance(required, list) or len(required) < 6 or len(required) != len(set(required)):
        stop("RIGHTS_AND_EVIDENCE_GATES_MISSING")
    for name in ("DATA_RIGHTS_AND_DEIDENTIFICATION", "INDEPENDENT_PATIENT_GROUP_REGISTRY",
                 "FROZEN_EXTERNAL_HELDOUT_TEST", "SIGNED_GPU_CAPACITY_BUDGET"):
        if name not in required:
            stop("RIGHTS_AND_EVIDENCE_GATES_MISSING")
    proto = closed(doc["validation"], VALIDATION_KEYS, "EVAL_SCHEMA")
    if (proto["partitions"] != ["training", "validation", "heldout_test"]
            or proto["unit"] != "independent_specimen_or_patient"
            or proto["site_stain_shift_required"] is not True
            or proto["physical_microns_per_pixel_required"] is not True
            or type(proto["instance_iou_threshold_strictly_above"]) not in {float, int}
            or proto["instance_iou_threshold_strictly_above"] != 0.5
            or proto["reported_scores"] != "NONE_CURRENTLY_MEASURED_BY_SZL"):
        stop("SCIENTIFIC_SPLIT_OR_SCALE_GATE")
    sample = closed(proto["bootstrap"],
                    {"unit", "replicates", "seed", "minimum_n"}, "BOOTSTRAP_SCHEMA")
    if (sample["unit"] != "specimen" or type(sample["replicates"]) is not int
            or sample["replicates"] != 1000 or type(sample["seed"]) is not int
            or not 0 <= sample["seed"] < 2**32
            or type(sample["minimum_n"]) is not int or sample["minimum_n"] < 5):
        stop("BOOTSTRAP_SPECIMEN_GATE")
    measured = closed(proto["metric_domains"], set(METRICS), "METRIC_DOMAINS")
    if any(set(values) != allowed for name, allowed in METRICS.items()
           for values in [measured[name]] if isinstance(values, list)) or any(
           not isinstance(measured[name], list) for name in METRICS):
        stop("TASK_METRIC_CONTRACT")
    sources = doc["sources"]
    arms = doc["arms"]
    if (not isinstance(sources, list) or len(sources) != 15
            or not isinstance(arms, list) or len(arms) != 9):
        stop("SOURCE_OR_ABLATION_COVERAGE")
    by_id = {}
    for item in sources:
        closed(item, SOURCE_KEYS, "SOURCE_SCHEMA")
        source_id, repo, revision = item["id"], item["repo"], item["sha"]
        if (not isinstance(source_id, str) or not ID_RE.fullmatch(source_id)
                or source_id in by_id or not isinstance(repo, str)
                or not SOURCE_RE.fullmatch(repo) or not isinstance(revision, str)
                or not SHA_RE.fullmatch(revision)):
            stop("DUPLICATE_OR_UNPINNED_SOURCE")
        url = item["paper"]
        if not isinstance(url, str):
            stop("PAPER_REFERENCE_INVALID")
        parts = urlsplit(url)
        if (parts.scheme != "https" or parts.hostname not in DOMAINS
                or parts.username or parts.password or parts.fragment or parts.query):
            stop("PAPER_REFERENCE_INVALID")
        for key in ("idea", "license_review"):
            if not isinstance(item[key], str) or not 4 <= len(item[key]) <= 260:
                stop("UNREVIEWED_RESEARCH_RIGHTS")
        by_id[source_id] = item
    by_arm = {}
    candidates = []
    for item in arms:
        closed(item, ARM_KEYS, "ARM_SCHEMA")
        key, task = item["id"], item["task"]
        if (not isinstance(key, str) or not ID_RE.fullmatch(key)
                or key in by_arm or task not in METRICS):
            stop("ARM_ID_OR_TASK")
        if (not isinstance(item["source_ids"], list)
                or not 1 <= len(item["source_ids"]) <= 5
                or len(item["source_ids"]) != len(set(item["source_ids"]))
                or any(source not in by_id for source in item["source_ids"])):
            stop("UNREFERENCED_LITERATURE")
        if item["baseline"] is not None:
            if item["baseline"] not in by_arm:
                stop("BASELINE_IS_NOT_PREVIOUS")
            if by_arm[item["baseline"]] != task:
                stop("CROSS_TASK_BASELINE")
        if (not isinstance(item["metrics"], list) or not item["metrics"]
                or len(item["metrics"]) != len(set(item["metrics"]))
                or any(metric not in METRICS[task] for metric in item["metrics"])):
            stop("CROSS_TASK_METRICS")
        if (item["source_only"] is not True or item["training_authorized"] is not False
                or item["model_weights_admitted"] is not False
                or item["benchmarked"] is not False):
            stop("UNQUALIFIED_MODEL_OR_JOB")
        if not isinstance(item["objective"], str) or not 10 <= len(item["objective"]) <= 500:
            stop("OBJECTIVE_UNAVAILABLE")
        by_arm[key] = task
        candidates.append({
            "id": key, "task": task, "compared_with": item["baseline"],
            "metrics": item["metrics"], "hypothesis": item["objective"],
            "source_bindings": [
                {"id": name, "git_commit": by_id[name]["sha"],
                 "read_only_url": f"https://github.com/{by_id[name]['repo']}/tree/{by_id[name]['sha']}"}
                for name in item["source_ids"]],
            "weights_or_data_approved": False,
            "training_steps_completed": 0,
            "evaluation_state": "NOT_EXECUTED",
            "action_authorized": False,
        })
    return {
        "schema": OUTPUT_SCHEMA,
        "status": "SOURCE_PINNED_ADVISORY_ONLY",
        "literature_count": len(by_id),
        "candidate_count": len(candidates),
        "tasks": sorted(METRICS),
        "registry_jcs_sha256": hashlib.sha256(rfc8785.dumps(doc)).hexdigest(),
        "experiment_candidates": candidates,
        "scientific_protocol": proto,
        "open_gates": required,
        "training_authorized": False,
        "model_publication_authorized": False,
        "model_weights_downloaded": False,
        "signatures_verified": False,
        "benchmark_results": None,
        "clinical_use_authorized": False,
        "caveat": "Research citations and Git hashes are not model rights, downloaded weights, patient disjointness or measured training.",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Compile a research-only histology experiment plan")
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.registry.is_symlink() or not args.registry.is_file():
            stop("SOURCE_FILE_UNAVAILABLE")
        with args.registry.open("rb") as source:
            raw = source.read(MAX_BYTES + 1)
        result = study_plan(raw)
        data = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
        if args.out.is_symlink():
            stop("OUTPUT_SYMLINK_FORBIDDEN")
        with args.out.open("xb") as target:
            target.write(data)
            target.flush()
            os.fsync(target.fileno())
        print(json.dumps({"status": result["status"],
                          "literature_count": result["literature_count"],
                          "candidate_count": result["candidate_count"],
                          "output_sha256": hashlib.sha256(data).hexdigest(),
                          "training_started": False}, sort_keys=True))
        return 0
    except Hold as exc:
        print(json.dumps({"status": "HOLD", "reason": str(exc),
                          "training_started": False}, sort_keys=True))
        return 2
    except (OSError, FileExistsError):
        print(json.dumps({"status": "HOLD", "reason": "FILE_OR_OUTPUT_UNAVAILABLE",
                          "training_started": False}, sort_keys=True))
        return 2


if __name__ == "__main__":
    sys.exit(main())
