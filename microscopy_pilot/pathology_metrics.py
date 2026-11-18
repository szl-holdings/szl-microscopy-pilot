"""Bounded, original CPU-only histopathology instance scores and uncertainty.

No third-party model or weight imports; test fixtures are synthetic masks.
A score alone is never evidence of an independently evaluated model.
"""
from __future__ import annotations

import math
import random
import re
from collections import Counter
from fractions import Fraction

MAX_PIXELS = 1_000_000
MAX_INSTANCES = 100_000
HMAC_RE = re.compile(r"[0-9a-f]{64}\Z")


class MetricHold(ValueError):
    """A closed, non-private metric admission refusal."""


def instance_scores(truth: list[list[int]], predicted: list[list[int]], *,
                    mpp_x: float, mpp_y: float) -> dict:
    """Compute panoptic quality and foreground Dice at explicit physical scale.

    Instance masks have integer labels; 0 denotes background. Matching is
    strictly IoU > 0.5, so disjoint labeled masks have unique assignments.
    """
    if any(type(value) not in (float, int) or not math.isfinite(value)
           or not 0.01 <= value <= 100 for value in (mpp_x, mpp_y)):
        raise MetricHold("INVALID_PHYSICAL_SCALE")
    if (not isinstance(truth, list) or not isinstance(predicted, list)
            or not truth or len(truth) != len(predicted)
            or not isinstance(truth[0], list)):
        raise MetricHold("INVALID_RASTER")
    height, width = len(truth), len(truth[0])
    if not 0 < width * height <= MAX_PIXELS:
        raise MetricHold("PIXEL_BUDGET")
    a, b, both = Counter(), Counter(), Counter()
    foreground_intersection = 0
    for original_row, test_row in zip(truth, predicted, strict=True):
        if (not isinstance(original_row, list) or not isinstance(test_row, list)
                or len(original_row) != width or len(test_row) != width):
            raise MetricHold("RAGGED_OR_MISALIGNED_RASTER")
        for gt, pr in zip(original_row, test_row, strict=True):
            if any(type(x) is not int or not 0 <= x < 2**31 for x in (gt, pr)):
                raise MetricHold("INVALID_INSTANCE_LABEL")
            if gt:
                a[gt] += 1
            if pr:
                b[pr] += 1
            if gt and pr:
                both[(gt, pr)] += 1
                foreground_intersection += 1
    if len(a) > MAX_INSTANCES or len(b) > MAX_INSTANCES:
        raise MetricHold("INSTANCE_COUNT_LIMIT")
    matched = []
    taken_gt, taken_pr = set(), set()
    for (gt, pr), intersect in both.items():
        iou = Fraction(intersect, a[gt] + b[pr] - intersect)
        if iou > Fraction(1, 2):
            if gt in taken_gt or pr in taken_pr:
                raise MetricHold("NON_UNIQUE_MATCH")
            taken_gt.add(gt)
            taken_pr.add(pr)
            matched.append(iou)
    tp, fp, fn = len(matched), len(b)-len(matched), len(a)-len(matched)
    object_denom = 2*tp + fp + fn
    foreground_denom = sum(a.values()) + sum(b.values())
    return {
        "schema": "szl.pathology-instance-metrics/v1",
        "pixel_count": width*height,
        "gt_instances": len(a), "pred_instances": len(b),
        "true_positives": tp, "false_positives": fp, "false_negatives": fn,
        "pq": float(2*sum(matched, Fraction()) / object_denom)
              if object_denom else None,
        "detection_f1": float(Fraction(2*tp, object_denom))
                        if object_denom else None,
        "segmentation_quality": float(sum(matched, Fraction()) / tp)
                                if tp else None,
        "foreground_dice": float(Fraction(2*foreground_intersection,
                                          foreground_denom))
                           if foreground_denom else None,
        "gt_area_um2": sum(a.values())*mpp_x*mpp_y,
        "pred_area_um2": sum(b.values())*mpp_x*mpp_y,
        "match_rule": "IoU_strictly_greater_than_0.5",
        "independent_model_evaluation_proved": False,
        "clinical_use_authorized": False,
    }


def paired_specimen_bootstrap(rows: list[dict], *, seed: int = 20261118,
                              replicates: int = 1000) -> dict:
    """Paired PQ-delta percentile interval, resampling unique specimens.

    A pseudonymous specimen string is not proof of independent patients.
    Missing metrics fail; never fill them with zero or inherit a prior score.
    """
    if (not isinstance(rows, list) or not 5 <= len(rows) <= 10_000
            or type(seed) is not int or not 0 <= seed < 2**32
            or type(replicates) is not int or replicates != 1000):
        raise MetricHold("SPECIMEN_BOOTSTRAP_BUDGET")
    observed_ids = set()
    deltas = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "specimen_hmac_sha256", "candidate_pq", "baseline_pq"
        }:
            raise MetricHold("SPECIMEN_SAMPLE_SCHEMA")
        identity = row["specimen_hmac_sha256"]
        if (not isinstance(identity, str) or not HMAC_RE.fullmatch(identity)
                or identity in observed_ids):
            raise MetricHold("DUPLICATED_OR_INVALID_SPECIMEN_ID")
        observed_ids.add(identity)
        baseline, candidate = row["baseline_pq"], row["candidate_pq"]
        if (any(type(x) not in (int, float) or not math.isfinite(x)
                or not 0 <= x <= 1 for x in (baseline, candidate))):
            raise MetricHold("PQ_VALUE_UNAVAILABLE")
        deltas.append(float(candidate-baseline))
    rng = random.Random(seed)
    n = len(deltas)
    sample_means = sorted(
        sum(deltas[rng.randrange(n)] for _ in range(n))/n
        for _ in range(replicates)
    )
    return {
        "schema": "szl.paired-specimen-pq-bootstrap/v1",
        "unit": "declared_specimen",
        "paired_specimens": n,
        "tiles_as_independent_samples": False,
        "mean_delta_pq": sum(deltas)/n,
        "percentile_95_interval": {
            "lower": sample_means[24],
            "upper": sample_means[974],
        },
        "seed": seed, "replicates": replicates,
        "actual_patient_distinctness_proved": False,
        "external_test_set_verified": False,
        "superiority_qualified": False,
        "training_started": False,
    }
