# HistoForge → SZL evidence-gated histology training intake

**Status: SOURCE IMPLEMENTATION, OFFLINE INTAKE ONLY. Not an executed training
run, GPU benchmark, model evaluation, model download, HistoForge extension,
clinical device, authorized upload or deployment.**

This is the first governed join between a new research source and the existing
SZL scientific/training estate. It belongs in the existing
`szl-holdings/szl-microscopy-pilot`, not in a competing paid
GPU scheduler. We did not fork HistoForge, install QuPath, scrape patient images
or mirror third-party model weights.

## What HistoForge actually offers

- **QuPath extension:** image/ROI labeling, correction, cell and region
  segmentation, training preparation and measurement. GPL-3.0; compatible
  with QuPath 0.6/0.7. The extension is not imported or statically linked
  into the permissive SZL Python codebase.
- **HistoForge backend:** Python/Modal orchestration, guarded upload,
  training jobs and dataset lifecycle. Apache-2.0. Python backend uses
  strict JCS (RFC 8785) scientific dataset identity.
- **HistoForge models:** DINOv3 ViT-L/16 plus Cellpose, HoVer-Net or dense
  semantic heads, Apache-2.0 with retained Cellpose BSD-3-Clause notices.
  These are *implementations*, not a source of unrestricted model weights.
- **Model cards:** HistoForge's DINOv3 starter and Lizard modules returned
  gated status and `license:other` in observed Hugging Face metadata.
  **Do not fetch, retrain, redistribute, deploy or publish those weights
  until their actual license terms, access rights and downstream-use
  permissions are independently reviewed.** Changing a .safetensors
  filename does not create model compatibility.
- **Data boundary:** analysis or fine-tuning with HistoForge normally uploads
  image data to the user's *own* Modal account; an ROI selection may upload
  the whole slide and embedded metadata. Training uploads remain stored
  until deleted. No Modal account or paid job is created by SZL's adapter.

Citable upstream source revisions at integration review time
(November 18, 2026 UTC):

| Component | Exact GitHub source |
| --- | --- |
| Backend | `HistoForge/histoforge-backend@ec0a838dcc875d5c01c68c6c585747bc615fd6cf` |
| Model implementations | `HistoForge/histoforge-models@7c299f75d0dc15a10318b49c85285f490fa476e0` |
| QuPath extension | `HistoForge/histoforge-qupath-extension@147a0813cd2682a8a38aa88f31c4c38f35c10b2a` |

Read only as source; versions can advance and require a new explicit review.

## Source-native input, not a fabricated interchange format

HistoForge's current training export is a ZIP containing a **JCS-canonical
`dataset.json` schema_version 2** with `dataset_digest`,
`task`, `artifacts`, `project_groups`, `images`, `rois`, and `files`.
The manifest declares an exact inventory of
`artifacts/<n>/origin.json`, an image source member
(`source.png`, `source.ome.tif`, or `source.tif`), and
`rois/<n>/roi.geojson` + `rois/<n>/objects.geojson`.
A companion HistoForge training split sidecar is
`{"schema_version":1,"dataset_digest":"…","split_plan":{…}}`.

Those fields are from the pinned backend source's
`histoforge/cloud/training/datasets/manifest.py`,
`archive.py`, `split.py` and `canonical_json.py`, not invented
project conventions. The source-owned `model_microns_per_pixel` matters:
a 100-pixel object at 0.5 µm/px covers **25 µm²**. Comparing raw
pixel areas at different resolutions is scientifically invalid.

SZL additionally needs one small, **local** declaration JSON recording the
input paths and hashes, exact HistoForge revisions, a SHA-pinned intended
Hugging Face model identity, and *untrusted* permission claims. These
declarations cannot themselves prove data rights.

```json
{
  "schema": "szl.histoforge-intake-request/v1",
  "upstream": {
    "backend_commit": "ec0a838dcc875d5c01c68c6c585747bc615fd6cf",
    "models_commit": "7c299f75d0dc15a10318b49c85285f490fa476e0",
    "extension_commit": "147a0813cd2682a8a38aa88f31c4c38f35c10b2a"
  },
  "archive": {
    "path": "dataset.zip",
    "size_bytes": 123456,
    "sha256": "<64-lowercase-hex>"
  },
  "split_sidecar": {
    "path": "split.json",
    "size_bytes": 456,
    "sha256": "<64-lowercase-hex>"
  },
  "module": {
    "repo": "histoforge/starter-dinov3-cellpose",
    "revision": "<exact-40-hex-Hub-commit>",
    "model_id": "objects",
    "weight_terms": "UNVERIFIED"
  },
  "permissions": {
    "local_training_allowed": false,
    "cloud_upload_allowed": false,
    "specimen_identity_reviewed": false,
    "publication_allowed": false
  }
}
```

This is an **example contract**, not a claim that an actual dataset/model is
available at those placeholder digests or that current source permits any use.

## One-shot admission, no cloud and no model execution

```sh
uv run --frozen python -m microscopy_pilot histoforge-admit \
  --request /private/research/intake.json \
  --out /private/research/intake-evidence.json
```

The same pure Python adapter can be invoked as
`python -m microscopy_pilot.histoforge_intake` with those flags. No
network request or training can be triggered by either command.
Output is exclusive-create with a SHA-256 of its bytes; missing/wrong
data fails with an explicit HOLD code and **never** falls back to a
previously passing manifest.

Bounded protective checks:

1. Exact vetted HistoForge source commits; a newer upstream source is a
   review gate, not an automatically accepted API change.
2. Exact local archive/sidecar byte counts, SHA-256, JCS canonical
   scientific preimage digest; strict reject duplicate JSON object keys.
3. Closed ZIP member list, relative ASCII paths, no traversal/symlink,
   no encryption, unsupported compression or zip-bomb expansion;
   streaming SHA-256 of **all** declared members. **No extraction**.
4. Source image digests sorted and unique, ROI identities and artifact
   links valid, complete file closure and nonzero physical MPP.
5. Dataset-bound split sidecar with training **and** validation groups.
   Only `project_group` splitting is admitted to this SZL lane; ROI-
   and image-based splitting can leak tissue from the same specimen
   into both partitions, so they stop rather than silently pass.
6. Privately sensitive group IDs and ROI paths are never printed in
   the evidence output. Explicit self-declared permission flags remain
   **untrusted**, even if all set to true.

Pilot caps: archive and extracted member totals ≤128 MiB; each member
≤64 MiB; ≤4,000 members; ≤512 images; ≤2,000 ROIs; ≤512 groups.
An ordinary whole slide can exceed those caps and is **HELD** for a
separately authorized streaming/paged ingestion design. The adapter does
not bypass the cap by uploading the slide to a different cloud.

## Critical proof distinctions

The successful receipt says
`SOURCE_BYTES_AND_SPECIMEN_SPLIT_CHECKED`. It **does not say**
`training_ready`, `trained`, `qualified`, or `licensed`.

The following gates remain mandatory and are not implemented here:

- Run HistoForge's **exact pinned full scientific validator** against
  source slide pixels, ROI/annotation geometry, label completeness,
  calibration, and actual capability profile. An archive with valid hashes
  can still contain scientifically invalid or synthetic pixels.
- Verify each specimen/patient grouping against an independent
  deidentified register; distinct image hashes **do not prove** distinct
  patients. A self-declared group string is not a patient-identity proof.
- Review dataset and subject rights, image metadata, export destinations,
  the actual model card, architecture, pretraining and redistribution
  terms. HistoForge model cards with gated `license:other` cannot be
  treated as Apache-2.0 weights.
- Define a withheld **test** cohort separate from both training and
  validation. Record per-tissue/stain/site scores, cell-level F1,
  segmentation Dice/PQ/IoU as applicable, inference cost/latency,
  confusion matrices and **patient-level** uncertainty; specify
  denominators, versions, baselines and scientifically meaningful units.
- Before **any** GPU execution, obtain written data-use approval and
  compute budget, bind the exact model+dataset+trainer to SZL's existing
  **DSSE-signed** `szl-gpu-bridge` owner queue, and get explicit
  authorization. This module never writes to `queue/pending/`.
- Training status, integrity-verified checkpoint, evaluation,
  Hugging Face publication, Cloudflare-hosted service and production
  qualification are distinct gates. Never publish images, private
  artifacts or claims merely because a CLI or CI job passed.

## Architecture of the future admitted lane

```text
QuPath (GPL, outside SZL Python process)
    └─ HistoForge export: exact dataset ZIP + split sidecar
          └─ szl-microscopy-pilot histoforge-admit (CPU, no cloud)
                ├─ verified local archive bytes + JCS identity
                ├─ specimen group partition invariants
                └─ private refusal/receipt + outstanding gates
                      └─ [future independently reviewed rights and science gate]
                            └─ szl-gpu-bridge owner-controlled DSSE job
                                  └─ Owner GPU (A6000) or approved paid job
                                        └─ trained weights + selected model + optimizer
                                              └─ independent frozen-test evaluation
                                                    └─ governed HF card/private/public
                                                          └─ Cloudflare runtime ONLY if qualified
```

Nothing below the review gate exists merely because this source is merged.
That prevents an externally trained histology model from silently inheriting
language-model evaluations, weights or licenses.

## Upstream citations and licenses

- https://histoforge.github.io/docs/reference/data-and-privacy.html
- https://histoforge.github.io/docs/training/prepare-data.html
- https://histoforge.github.io/docs/training/review-result.html
- https://histoforge.github.io/docs/reference/recipe.html
- https://github.com/HistoForge/histoforge-backend/tree/ec0a838dcc875d5c01c68c6c585747bc615fd6cf
- https://github.com/HistoForge/histoforge-models/tree/7c299f75d0dc15a10318b49c85285f490fa476e0
- https://github.com/HistoForge/histoforge-qupath-extension/tree/147a0813cd2682a8a38aa88f31c4c38f35c10b2a

HistoForge's software citation: HistoForge contributors (2026),
*HistoForge* (computer software); cite QuPath and each model/dataset
separately. **Non-clinical research use only** until a validated,
authorized clinical path independently exists.
