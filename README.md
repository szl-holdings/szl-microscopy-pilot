# Microscopy image QC pilot

A small, CPU-only example of how to attach a checkable explanation to an image-quality measurement. It reads two images, verifies their recorded byte hashes, measures a few simple pixel properties, and writes a JSON report with evidence, caveats, and a pending human decision. The example uses a matched field from [BBBC006v1](https://bbbc.broadinstitute.org/BBBC006), an openly released U2OS microscopy image set. This is a worked quality-control example, not a benchmark result or a biological finding.

## Five-minute quickstart

From the repository root, with [uv](https://docs.astral.sh/uv/) available:

```sh
uv --cache-dir .uv-cache run --locked python -m microscopy_pilot run --manifest examples/bbbc006v1/manifest.json --out report.json
```

The command verifies the local source files before measurement and writes `report.json`. It uses Python 3.10 or newer and Pillow 12.3.0. To check the source files without producing a report:

```sh
uv --cache-dir .uv-cache run --locked python -m microscopy_pilot verify --manifest examples/bbbc006v1/manifest.json
```

See the [five-minute walkthrough](docs/WORKED_EXAMPLE.md) and its [readable report](docs/worked_report.html). The machine-readable output follows [the explanation schema](schemas/explanation.schema.json). The command runs locally; it does not call an atlas or model API.

## What the report says

The report keeps each kind of evidence visible:

| Field | Meaning |
| --- | --- |
| `dataset` and `images` | Accession, source page, citation, source URLs, byte counts, SHA-256 hashes, and measured image properties. |
| `method` | The exact algorithm version, parameters, and known limits. |
| `comparison` | Whether the two images can be compared and, if so, which has the larger measured focus proxy. |
| `explanation` | A bounded image-quality claim, supporting metric pointers, counter-evidence, and uncertainty. |
| `human_decision` | `pending`. A researcher can decide `keep`, `exclude`, `reacquire`, or `uncertain` after reviewing the images and experiment context. |
| `biological_assessment` | `abstain`. Image-quality measurements do not establish phenotype, diagnosis, cell identity, or assay effect. |
| `atlas` and `specialist_model` | `not_configured`, with empty evidence. No atlas match or specialist-model result is implied. |

The calculation converts pixel values to a grayscale intensity in `[0, 1]`. It records dimensions, mean intensity, fractions within one code value of the encoded dark and bright endpoints, and a four-neighbor Laplacian variance. The comparison uses a dimensionless `focus_proxy`: mean squared interior four-neighbor Laplacian divided by squared mean intensity. Higher values indicate more high-frequency image structure under this definition. They do not by themselves prove that an image is in focus. The near-endpoint fractions describe encoded pixels; for 16-bit TIFF they do **not** establish detector saturation without the acquisition range.

Comparisons require the same recorded field, channel, and dimensions. Recorded exposure, gain, or processing settings must agree when provided for both images. Unrecorded settings are called out as uncertainty. Flat texture, mismatched fields or channels, gain or normalization changes, and deterministic image noise can confound a focus proxy. The report offers a measurement and a reasoned review aid, never an automatic exclusion.

## Example provenance and use

The worked images come from [BBBC006, version 1](https://bbbc.broadinstitute.org/BBBC006). The source describes a z-stack of U2OS cells, with z = 16 at the optimal focal plane and z = 11–23 classified in focus by an expert. Our manifest records the chosen files, their archive URLs, local byte counts, hashes, and field metadata. The raw 16-bit TIFFs are a single matched field, not the entire data set. The example does not train a model, segment cells, or validate a general-purpose focus classifier.

Please cite the source as recommended by the collection: “We used the image set BBBC006v1 from the Broad Bioimage Benchmark Collection [Ljosa et al., *Nature Methods*, 2012, [doi:10.1038/nmeth.2083](https://doi.org/10.1038/nmeth.2083)].” The [BBBC006 page](https://bbbc.broadinstitute.org/BBBC006) links the images and records Anne Carpenter's copyright waiver for the U2OS images and ground truth under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). The original code and documentation in this repository are under [MIT](LICENSE). The dataset's CC0 status and scientific citation are separate from this repository's software license; neither implies endorsement by Broad or the image contributors.

## Reproduce or adapt the example

`examples/bbbc006v1/manifest.json` is the input record. Preserve its source and license fields, exact byte counts, and SHA-256 hashes when sharing a report. A changed or damaged file must fail verification rather than silently producing new measurements. The run verifies local bytes; source URLs and acquisition metadata are recorded from the manifest and are not independently checked against the remote archive on each run. The report contains the method version, decoder, and parameters so a later run can be compared with the same algorithm. For another sample, provide verified metadata and permission, then use the same schema to keep observations separate from biological interpretation.

The checked-in [example report](examples/bbbc006v1/report.json) is the reference output. It retains its original Pillow 11.3.0 decoder provenance. The Pillow 12.3.0 compatibility test verifies identical image measurements and report fields except for the newly reported decoder version; it does not rewrite that historical report. To regenerate the static HTML view from that report and the original TIFFs:

```sh
uv --cache-dir .uv-cache run --locked python -m tools.build_worked_report
```

The HTML embeds small display-only PNG previews made with one shared linear scale; the code measures the original TIFF bytes.

The tests include missing metadata, damaged bytes or hash mismatch, unsupported biological claims, and absent atlas evidence. Run them from the repository root with:

```sh
uv --cache-dir .uv-cache run --locked python -m unittest discover -s tests
```

No patient data, paid model service, credential, or Docker environment is required.
