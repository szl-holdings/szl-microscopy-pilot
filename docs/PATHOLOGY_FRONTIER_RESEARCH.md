# SZL pathology frontier: comparative study, original math and training admission

**Status:** original source-tested experiment design in the existing
`szl-microscopy-pilot`. **No GPU training started, no external weights or
pathology slides downloaded, no Modal/SaladCloud job, no model benchmark
completed, no clinical qualification.** This is a *research-to-training*
handoff, not a new scheduler or a model publication.

## Evidence instead of leaderboard imitation

HistoForge provides QuPath annotation, Python/Modal model training and
native JCS (RFC 8785) dataset-v2 / split-v1 contracts. Its available
DINOv3+Cellpose/HoVer-Net heads are **nuclei segmentation** architectures.
The next step is an equal-data, rights-cleared *paired experiment*, not
to replace that stack with a large slide classifier.

Recent multi-task comparisons in Nature Communications (2026) and
Scientific Reports (2026) find that Virchow2, Prov-GigaPath, UNI2,
H-optimus-0 and other encoders perform differently by tissue, endpoint,
and training domain. The [2026 robustness study](https://www.nature.com/articles/s41467-026-73923-2)
warns that standard-test performance does not guarantee resilience to
new sites and staining conditions. **No one model is universally best,
and an external AUROC is not an SZL nucleus-PQ result.**

Primary comparison: [Nature Communications 2026](https://www.nature.com/articles/s41467-026-76004-6);
[2026 foundation-model tradeoff study](https://www.nature.com/articles/s41598-026-69731-9).

## Researchers, publications and exact source

The machine-readable [literature and experiments
registry](../research/pathology_frontier_v1.json) contains **15 full Git
commit pins and nine non-executable experiments**. Pinning repository
code does **not** pin a separate Hub model card, secure the weights,
grant permission to redistribute a model or prove source authenticity.

| Leader | Publication, source and method to study | Adoption boundary |
|---|---|---|
| HistoForge / QuPath | [Documentation](https://histoforge.github.io/docs/), [backend](https://github.com/HistoForge/histoforge-backend), [models](https://github.com/HistoForge/histoforge-models), [QuPath extension](https://github.com/HistoForge/histoforge-qupath-extension) | Existing SZL JCS data intake; QuPath GPL-3.0 code stays out of the MIT Python process; full slides may be uploaded to the user's own Modal workspace, **not by this code**. |
| Meta FAIR / DINOv3 | [2025 report](https://ai.meta.com/research/publications/dinov3/), [official PyTorch](https://github.com/facebookresearch/dinov3) | Dense SSL features and efficient spatial adapters; inspect Meta's **separate commercial-license conditions** before any weights. |
| TIO-IKIM / CellViT++ | [2026 journal](https://doi.org/10.1016/j.cmpb.2025.109206), [official code](https://github.com/TIO-IKIM/CellViT-plus-plus) | Lightweight frozen-feature cell classification; **Commons Clause restriction**, so no automatic code copying/commercial exploitation. |
| Pape Lab / PathoSAM | [PMLR 2026](https://proceedings.mlr.press/v301/griebel26a.html), [MIT source](https://github.com/computational-cell-analytics/patho-sam) | Interactive nucleus masks, human correction time and out-of-domain masks; weight/dataset terms separate. |
| Van Valen / CellSAM | [Nature Methods 2025](https://doi.org/10.1038/s41592-025-02879-w), [code](https://github.com/vanvalenlab/cellSAM) | CellFinder+SAM prompted masks across modalities; model and dataset rights independent. |
| Stringer/Pachitariu / Cellpose-SAM & Cellpose4 | [2025 preprint](https://www.biorxiv.org/content/10.1101/2025.04.28.651001v1), [official code](https://github.com/MouseLand/cellpose) | Flow-head, 2026 DINOv3-derived CP models, human-in-the-loop; published pretraining models/data **CC-BY-NC**. |
| Vu/Rajpoot / HoVer-Net | [Medical Image Analysis 2019](https://doi.org/10.1016/j.media.2019.101563), [official code](https://github.com/vqdang/hover_net) | Horizontal/vertical nucleus displacement and instance typing; separate head ablation. |
| DKFZ / nnU-Net | [Nature Methods 2021](https://doi.org/10.1038/s41592-020-01008-z), [code](https://github.com/MIC-DKFZ/nnUNet) | Own-weight, train-from-scratch semantic segmentation baseline. |
| MONAI consortium | [Code](https://github.com/Project-MONAI/MONAI) | Deterministic transforms, loss/augmentation benchmarking; not automatic evidence of a pathology-trained model. |
| Warwick / TIAToolbox | [Communications Medicine 2022](https://doi.org/10.1038/s43856-022-00186-5), [code](https://github.com/TissueImageAnalytics/tiatoolbox) | WSI/ROI/tile/QuPath interoperability and bounded image readout. |
| MahmoodLab / UNI2 | [UNI repository](https://github.com/mahmoodlab/UNI) | Frozen ROI/slide features only after gated weights/rights review; **not** a nucleus instance metric. |
| Providence/Microsoft / Prov-GigaPath | [Nature 2024](https://doi.org/10.1038/s41586-024-07441-w), [GitHub](https://github.com/prov-gigapath/prov-gigapath) | Tissue/slide context and frozen tile embeddings, separate slide endpoint. |
| Paige / Virchow | [Nature Medicine 2024](https://doi.org/10.1038/s41591-024-03141-0), [SDK](https://github.com/Paige-AI/paige-ml-sdk) | Noncommercial research-restricted weights: no presumed commercial training or runtime permission. |

The nine [experiment arms](../research/pathology_frontier_v1.json)
stage: independent scratch baseline; HoVer offsets; Cellpose flow;
PathoSAM interactive human correction; CellSAM generalization; frozen
DINOv3 adapter; CellViT++ cell classification; stain/site-shift
semantic segmentation; and separate UNI2/GigaPath/Virchow slide-feature
comparison. All are `training_authorized=false` and have no scores.

## Original SZL mathematics implemented now

`microscopy_pilot/pathology_metrics.py` is original, Python-only
bounded code operating on caller-provided integer instance-label masks.
It imports **no** HistoForge, Cellpose or SAM code/weights.

For one-to-one nucleus matches where `IoU > 0.5`:

```text
PQ = Σ IoU(matched nuclei) / [TP + 0.5 × FP + 0.5 × FN]
detection_F1 = 2 × TP / [2 × TP + FP + FN]
Dice = 2 × intersection(foreground) / [GT_foreground + Pred_foreground]
area_um² = pixel_count × mpp_x_um × mpp_y_um
```

Use exact integer overlaps and `fractions.Fraction` to avoid
misclassifying an IoU exactly equal to the strict 0.5 threshold.
Two empty masks return an **unavailable denominator**, not an
invented perfect PQ. A merged object can have Dice 1 but PQ 0:
instance topology and foreground coverage are distinct.

The second original routine compares paired **specimen-level** PQ
differences, resampling N distinct pseudonymized specimens exactly
1,000 times with a fixed random seed and returning a 95% percentile
interval. Five distinct records are required for the *synthetic
contract*. It never prints specimen IDs or says HMAC syntax proves
distinct patients, test independence or superiority. Sample counts
and uncertainty are evidence; a synthetic interval is **not** a
biological or clinical benchmark.

**Deferred intentionally:** PQ on actual labeled pathology images,
per-class F1, multiscale morphology, measured stain shift, calibration
ECE, site/external testing, measured VRAM/energy/latency, conformal
abstention, and truly trained adapters. None of those results exists
merely because a source method or plan was added.

## How the plan joins existing training, without executing it

```text
QuPath / HistoForge on licensed site
  -> native HistoForge dataset-v2 ZIP + split-v1
  -> existing SZL histoforge-admit (bounded exact JCS/ZIP/ROI checks)
  -> [separate full HistoForge scientific validator + rights clearance]
  -> research/pathology_frontier_v1.json -- source-pinned hypotheses
  -> microscopy_pilot.pathology_frontier (non-executing experiment plan)
  -> [independent patient ID, licensed model weights, three-way test split]
  -> [written approval + source-bound DSSE szl-gpu-bridge authority]
  -> [actual GPU execution + checkpoint readback]
  -> [paired test results on frozen external cohort]
  -> [governed HF distribution + deployed runtime, only if qualified]
```

Generate a **non-executable** study plan:

```sh
uv run --frozen python -m microscopy_pilot.pathology_frontier \
  --registry research/pathology_frontier_v1.json \
  --out /private/research/pathology-frontier-plan.json
```

The compiler performs a bounded, duplicate-key-free read and refuses
missing specimen-level train/validation/held-out splits, incompatible
task metrics, paper URL/commit tampering, missing license/data gates,
or any attempted execution/publication claim. It emits exact source
refs, a canonical SHA-256 of the registry and open gates. SHA-256 here
is **not** a digital signature or proof that the referenced external
repository still exists. The output is exclusive-create; it is **not**
the executable schema accepted by the GPU bridge.

Reproduce without a model or GPU:

```sh
uv run --frozen python -m unittest discover -s tests -v
```

After **explicit** rights/consent, artifact, budget and benchmark
approval, the separately authorized SZL GPU owner should design a
new image-specific trainer, bind the exact output shape/loss and
calibration to a disjoint external test cohort, and record the true
train/evaluate/publish/deploy state transitions. H&E pathology and
the existing BBBC006 fluorescent image QC are **different modalities**;
never combine their scores.

**Publication rule:** a Github commit and a plan are not
training, weights, scientific evaluation, clinical qualification,
performance superiority or commercial permission.
