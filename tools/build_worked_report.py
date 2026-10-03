"""Build a self-contained HTML view of the verified BBBC006v1 QC report.

This script checks the checked-in JSON against a fresh local run, verifies the
two TIFF hashes, then embeds 8-bit *display-only* PNG previews. Both previews
use one linear scale: black is encoded value 0 and white is the pooled 99.9th
percentile of the two original TIFFs. Values above white clip in the preview.
All reported QC metrics are calculated from the original TIFFs, never previews.
"""

from __future__ import annotations

import base64
import hashlib
import html
import io
import json
from pathlib import Path
from string import Template

from PIL import Image

from microscopy_pilot.core import run


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "examples" / "bbbc006v1" / "manifest.json"
REPORT = ROOT / "examples" / "bbbc006v1" / "report.json"
OUTPUT = ROOT / "docs" / "worked_report.html"


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _preview(values: list[int], size: tuple[int, int], white: int) -> str:
    data = bytes(min(255, (value * 255 + white // 2) // white) for value in values)
    image = Image.frombytes("L", size, data)
    stream = io.BytesIO()
    image.save(stream, format="PNG", optimize=True)
    return base64.b64encode(stream.getvalue()).decode("ascii")


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    recorded = json.loads(REPORT.read_text(encoding="utf-8"))
    fresh = run(MANIFEST)
    if fresh != recorded:
        raise ValueError("Checked-in report differs from a fresh verified run")
    if len(recorded["images"]) != 2 or recorded["comparison"]["status"] != "proxy_ranked":
        raise ValueError("Worked HTML requires the verified ranked BBBC006 pair")

    original: list[tuple[dict, list[int], tuple[int, int]]] = []
    all_values: list[int] = []
    for item, measured in zip(manifest["images"], recorded["images"]):
        path = (MANIFEST.parent / item["path"]).resolve()
        raw = path.read_bytes()
        if len(raw) != item["bytes"] or hashlib.sha256(raw).hexdigest() != item["sha256"]:
            raise ValueError(f"Source bytes changed: {path.name}")
        if item["id"] != measured["id"]:
            raise ValueError("Manifest/report image order differs")
        with Image.open(path) as source:
            if source.mode not in {"I;16", "I;16L", "I;16B"}:
                raise ValueError("Expected 16-bit native TIFF")
            values = list(source.getdata())
            size = source.size
        if size != (measured["metrics"]["width"], measured["metrics"]["height"]):
            raise ValueError("Source/report dimensions differ")
        original.append((measured, values, size))
        all_values.extend(values)

    all_values.sort()
    white = all_values[int((len(all_values) - 1) * 0.999)]
    if white <= 0:
        raise ValueError("No display range in source pair")

    images_html: list[str] = []
    metrics_rows: list[str] = []
    sources_html: list[str] = []
    maximum_proxy = max(item["metrics"]["focus_proxy"] for item, _, _ in original)
    for item, values, size in original:
        m = item["metrics"]
        preview = _preview(values, size, white)
        z = _escape(item["z_plane"])
        image_id = _escape(item["id"])
        images_html.append(
            f'<figure class="image-card"><div class="image-top"><strong>{z}</strong>'
            f'<span>{_escape(item["field_id"])} · {_escape(item["channel"])}</span></div>'
            f'<img src="data:image/png;base64,{preview}" width="{size[0]}" height="{size[1]}" '
            f'alt="Grayscale display preview of the BBBC006v1 {z} TIFF, field a02_s1, nuclei channel">'
            f'<figcaption>Original: {m["width"]} × {m["height"]}, {m["bit_depth"]}-bit TIFF. '
            f'This preview uses the shared display scale below.</figcaption></figure>'
        )
        pct = round(m["focus_proxy"] / maximum_proxy * 100, 2)
        metrics_rows.append(
            f'<tr><th scope="row">{z}</th><td><strong>{m["focus_proxy"]:.12f}</strong>'
            f'<div class="bar" aria-hidden="true"><span style="width:{pct}%"></span></div></td>'
            f'<td>{m["laplacian_variance"]:.12g}</td>'
            f'<td>{m["mean_intensity"]:.12f}</td>'
            f'<td>{m["clipped_dark_fraction"]:.3g} / {m["clipped_bright_fraction"]:.3g}</td></tr>'
        )
        sources_html.append(
            f'<dt>{z} archive</dt><dd><a href="{_escape(item["source_url"])}">'
            f'{_escape(item["source_url"])}</a><br>'
            f'<code>{_escape(item["source_member"])}</code><br>'
            f'{item["bytes"]:,} bytes · SHA-256 <code>{_escape(item["sha256"])}</code></dd>'
        )

    support = "".join(
        f'<li><code>{_escape(ev["image_id"])}</code>: '
        f'{ev["metric"]} = <strong>{ev["value"]:.12f}</strong> '
        f'(<code>{_escape(ev["pointer"])}</code>)</li>'
        for ev in recorded["explanation"]["supporting_evidence"]
    )
    counter = "".join(
        f'<li>{_escape(ev["observation"])} '
        f'(<code>{_escape(ev["pointer"])}</code>)</li>'
        for ev in recorded["explanation"]["counter_evidence"]
    )
    template = Template("""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BBBC006v1 | Microscopy image QC pilot</title>
  <style>
    :root { color-scheme: light; font-family: system-ui, -apple-system, Segoe UI, sans-serif; line-height: 1.5; color: #172a35; background: #f3f6f6; }
    * { box-sizing: border-box; }
    body { margin: 0; }
    header { background: #123b44; color: #fff; padding: 2.3rem max(1.5rem, calc((100vw - 1120px) / 2)); }
    header p { color: #cce4e3; max-width: 790px; margin-bottom: 0; }
    h1 { font-size: clamp(1.7rem, 4vw, 2.55rem); margin: .25rem 0; letter-spacing: -.03em; }
    h2 { font-size: 1.25rem; margin: 0 0 .75rem; }
    h3 { font-size: 1rem; margin: 0 0 .45rem; }
    main { max-width: 1120px; margin: auto; padding: 1.5rem; }
    section { background: #fff; border: 1px solid #dce5e4; border-radius: 12px; margin: 0 0 1rem; padding: 1.35rem; }
    .eyebrow { text-transform: uppercase; letter-spacing: .1em; font-size: .75rem; font-weight: 700; color: #72cbc4; }
    .grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1rem; }
    .image-card { margin: 0; border: 1px solid #dce5e4; border-radius: 9px; overflow: hidden; background: #fafcfc; }
    .image-top { padding: .7rem .85rem; display: flex; gap: .75rem; align-items: baseline; justify-content: space-between; }
    .image-top strong { color: #0c5556; font-size: 1.15rem; }
    .image-top span { color: #4f6770; font-size: .75rem; overflow-wrap: anywhere; text-align: right; }
    img { width: 100%; height: auto; display: block; background: #090b0c; }
    figcaption { color: #415c65; padding: .7rem .85rem; font-size: .83rem; }
    .callout { border-left: 4px solid #149385; background: #effaf8; padding: .85rem 1rem; margin: .9rem 0; }
    .warning { border-left-color: #b77912; background: #fff8e9; }
    .pills { display: flex; flex-wrap: wrap; gap: .5rem; margin: .8rem 0; }
    .pill { background: #e9f0f1; color: #35505a; border-radius: 999px; padding: .3rem .7rem; font-size: .83rem; }
    .pill.current { background: #d9f3ed; color: #155f53; font-weight: 700; }
    table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
    th, td { text-align: left; border-bottom: 1px solid #e1e9e8; padding: .75rem .55rem; vertical-align: top; }
    thead th { color: #526b73; font-size: .8rem; }
    .scroll { overflow-x: auto; }
    .bar { height: 5px; background: #e3eceb; border-radius: 4px; margin-top: .4rem; width: 100%; }
    .bar span { display: block; height: 100%; background: #178f88; border-radius: 4px; }
    code { overflow-wrap: anywhere; font-size: .85em; }
    dl { display: grid; grid-template-columns: 8rem minmax(0, 1fr); gap: .7rem 1rem; }
    dt { font-weight: 700; }
    dd { margin: 0; overflow-wrap: anywhere; }
    .muted { color: #4d6670; font-size: .88rem; }
    a { color: #0a6d70; }
    footer { max-width: 1120px; padding: 0 1.5rem 2rem; margin: auto; color: #52666c; font-size: .86rem; }
    @media (max-width: 760px) { .grid { grid-template-columns: 1fr; } dl { grid-template-columns: 1fr; gap: .2rem; } dd { margin-bottom: .7rem; } }
  </style>
</head>
<body>
  <header>
    <div class="eyebrow">Microscopy image QC pilot · worked example</div>
    <h1>One BBBC006v1 field, two z-planes</h1>
    <p>Verified 16-bit nuclei-channel TIFFs from field a02_s1. A deterministic pixel measurement ranks the two focus proxies. A researcher reviews the images and decides what to do.</p>
  </header>
  <main>
    <section>
      <h2>Observed result</h2>
      <div class="callout"><strong>$claim</strong></div>
      <p class="muted">Comparison status: <code>$comparison_status</code>. $comparison_reason</p>
      <div class="pills"><span class="pill current">Human review required</span><span class="pill">Decision pending</span><span class="pill">Biological assessment: abstain</span><span class="pill">Atlas: not configured</span><span class="pill">Specialist model: not configured</span></div>
      <p class="muted">The BBBC006 source identifies z16 as the optimal plane and z11–23 as expert-classified in focus. That source annotation is context; this two-image measurement does not validate a general classifier.</p>
    </section>
    <section>
      <h2>Source images</h2>
      <div class="grid">$images_html</div>
      <p class="muted">Display-only transform: both original TIFFs use the same linear grayscale mapping, encoded value 0 → black and $display_white → white (the pooled 99.9th percentile); higher values are clipped for display. No crop, sharpening, denoising, or individual-image scaling. Measurements below use the untouched 16-bit pixels.</p>
    </section>
    <section>
      <h2>Measured image quality</h2>
      <div class="scroll"><table><thead><tr><th>Plane</th><th>Focus proxy<br><small>dimensionless</small></th><th>Laplacian variance<br><small>normalized intensity²</small></th><th>Mean intensity<br><small>0–1</small></th><th>Near encoded endpoints<br><small>dark / bright fraction</small></th></tr></thead><tbody>$metrics_rows</tbody></table></div>
      <p class="muted">Focus proxy = mean squared interior four-neighbor Laplacian / squared mean intensity. These fractions count pixels within one encoded value of either endpoint; they do not prove detector saturation.</p>
    </section>
    <section class="grid">
      <div><h2>Supporting evidence</h2><ul>$support</ul><p class="muted">Each JSON Pointer refers to the <a href="../examples/bbbc006v1/report.json">machine-readable report</a>.</p></div>
      <div><h2>Counter-evidence and limits</h2><ul>$counter</ul><p class="muted">$uncertainty</p></div>
    </section>
    <section>
      <h2>Researcher decision</h2>
      <p>Status: <strong>pending</strong>. After visual and assay-context review, a researcher may choose one option and record a rationale. The program does not choose for them.</p>
      <div class="pills"><span class="pill">Keep</span><span class="pill">Exclude</span><span class="pill">Reacquire</span><span class="pill">Uncertain</span></div>
      <div class="callout warning">A high-frequency proxy can rise with noise, sharpening, or changed gain or normalization. Nearly constant texture may make it uninformative. Field/channel mismatch prevents comparison. Exposure and gain equality are not independently verified in these source records.</div>
    </section>
    <section>
      <h2>Provenance</h2>
      <p><a href="https://bbbc.broadinstitute.org/BBBC006">Broad Bioimage Benchmark Collection BBBC006, version 1</a> · <a href="https://creativecommons.org/publicdomain/zero/1.0/">CC0 1.0 waiver</a>. Citation: $citation</p>
      <dl>$sources_html</dl>
      <p class="muted">Each TIFF was extracted from its named archive with a bounded byte-range request and verified by byte count and SHA-256. The repository manifest records the inclusive archive ranges. A local rerun verifies local bytes; it does not independently check the manifest's source URLs or acquisition metadata against the remote archive. Neither this repository nor this page implies endorsement by Broad or the image contributors.</p>
    </section>
  </main>
  <footer>Static view generated by <code>tools/build_worked_report.py</code> from the verified manifest and report. Source data, measurements, and limitations are available in the repository.</footer>
</body>
</html>
""")
    page = template.substitute(
        claim=_escape(recorded["explanation"]["claim"]),
        comparison_status=_escape(recorded["comparison"]["status"]),
        comparison_reason=_escape(recorded["comparison"]["reason"]),
        images_html="".join(images_html),
        display_white=str(white),
        metrics_rows="".join(metrics_rows),
        support=support,
        counter=counter,
        uncertainty=_escape(recorded["explanation"]["uncertainty"]["reason"]),
        citation=_escape(recorded["dataset"]["citation"]),
        sources_html="".join(sources_html),
    )
    OUTPUT.write_bytes(page.encode("utf-8"))
    print(f"{OUTPUT}: {len(page.encode('utf-8')):,} bytes; shared display white={white}")


if __name__ == "__main__":
    main()
