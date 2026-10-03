# Five-minute BBBC006v1 walkthrough

This example uses two native 16-bit TIFFs from one field and channel in the [BBBC006v1](https://bbbc.broadinstitute.org/BBBC006) z-stack. The source lists z = 16 as its optimal focal plane and z = 11–23 as the expert's in-focus range. The example compares a z = 16 image with a z = 00 image. That source annotation is context for reading the example; the script does not infer ground-truth focus labels for new images.

## Minute 1: inspect the source record

Open [`examples/bbbc006v1/manifest.json`](../examples/bbbc006v1/manifest.json). Confirm the BBBC accession and version, source page, CC0 link, file paths, matched field and channel, source archive URLs, exact sizes, and SHA-256 hashes. The two small TIFFs are included so the demo does not download the full collection.

## Minute 2: run the check

From the repository root:

```sh
uv --cache-dir .uv-cache run --locked python -m microscopy_pilot run --manifest examples/bbbc006v1/manifest.json --out report.json
```

The program first checks file presence, byte lengths, hashes, decoding, and required metadata. It then writes `report.json` locally. A checked-in [reference report](../examples/bbbc006v1/report.json) records this run. A mismatch is an error, not a low-quality image result. Source URLs and acquisition metadata come from the manifest; this command verifies local bytes, not the remote archive again.

## Minute 3: read the measurements

In `report.json`, find `images[*].metrics`. Dimensions and mean intensity help establish the input scale. The dark and bright fractions count pixels at the encoded endpoints; they do not by themselves show camera saturation. `laplacian_variance` is a diagnostic for fine spatial structure. `focus_proxy` divides mean squared interior four-neighbor Laplacian by squared mean intensity, making the ranking less sensitive to a uniform intensity scale. It is still sensitive to texture, processing, and noise.

The `comparison` section ranks the focus proxy only for images with the same recorded field, channel, and dimensions. If exposure, gain, or processing metadata are present, they must agree. Missing acquisition settings remain a stated limitation. Read `comparison.reason` before treating the numerical ranking as useful.

In the verified example, `focus_proxy` is **0.031563799621** for z16 and **0.017273580964** for z00, so z16 has the larger measured proxy. The collection's expert focal-plane annotation also identifies z16 as optimal and z00 as outside its in-focus range. This agreement for one field does not validate the proxy for other samples.

## Minute 4: follow the explanation

The `explanation.supporting_evidence` entries give image IDs, measured values, source URLs, and JSON Pointers back to the measurements. `counter_evidence` points to limitations or observations that could weaken the claim. `uncertainty` states what remains unknown. The claim is confined to measured image quality; it says nothing about cell phenotype, disease, or treatment response. `biological_assessment.status` is `abstain`. `atlas.status` and `specialist_model.status` are `not_configured` with no evidence.

## Minute 5: make the human decision

Look at both images and consider the assay protocol and whether the field is usable. A researcher may choose **keep**, **exclude**, **reacquire**, or **uncertain**, and record a rationale outside the automatically generated report. The software leaves `human_decision.status` at `pending` and never excludes an image on its own.

For a readable view of the example, open [`worked_report.html`](worked_report.html). It embeds display-only previews from the original TIFFs, using a shared linear mapping from encoded 0 to the pair's pooled 99.9th-percentile value of 1110; values above 1110 clip in the preview. The preview transform has no effect on the raw-pixel measurements. The static page, JSON report, and manifest can be regenerated and checked locally.

### Why the result is limited

This is one matched field, not a validation study. Nearly constant texture may yield an unstable or uninformative focus score. Different fields, channels, gain, normalization, or processing can break a comparison. Noise may increase the proxy even when optical focus has not improved. The original 16-bit encoding alone does not reveal detector saturation range. The result should be checked by a microscopy researcher before any quality-control action.
