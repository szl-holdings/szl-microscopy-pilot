"""Fetch two bounded, verified BBBC006v1 TIFFs without downloading their ZIPs.

The byte ranges below are the raw DEFLATE payloads of one ZIP member per
archive. The Broad Institute's BBBC006 page documents the dataset and CC0
waiver: https://bbbc.broadinstitute.org/BBBC006

Run from any directory with Python 3.10+::

    python tools/fetch_native_example.py
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import re
import tempfile
from dataclasses import dataclass
from urllib.request import HTTPRedirectHandler, Request, build_opener
import zlib


MAX_COMPRESSED_BYTES = 512_000
MAX_UNCOMPRESSED_BYTES = 512_000
CONTENT_RANGE_RE = re.compile(r"bytes (\d+)-(\d+)/(\d+)")


@dataclass(frozen=True)
class Member:
    url: str
    member: str
    start: int
    end: int
    compressed_bytes: int
    uncompressed_bytes: int
    crc32: int
    sha256: str

    @property
    def filename(self) -> str:
        return self.member.rsplit("/", 1)[-1]


MEMBERS = (
    Member(
        url="https://data.broadinstitute.org/bbbc/BBBC006/BBBC006_v1_images_z_16.zip",
        member="BBBC006_v1_images_z_16/mcf-z-stacks-03212011_a02_s1_w166fea8cb-8e72-4999-9f72-ee77bfdac65f.tif",
        start=224_354_594,
        end=224_730_736,
        compressed_bytes=376_143,
        uncompressed_bytes=385_490,
        crc32=4_168_770_214,
        sha256="2808dd1f582832f8c2e906b8bd6316e1891acf4f69d577ae9d22f4aaa1ef82c5",
    ),
    Member(
        url="https://data.broadinstitute.org/bbbc/BBBC006/BBBC006_v1_images_z_00.zip",
        member="BBBC006_v1_images_z_00/mcf-z-stacks-03212011_a02_s1_w1e1ad67bb-1e50-4666-a01e-cdadc4cee89d.tif",
        start=347_000_250,
        end=347_417_924,
        compressed_bytes=417_675,
        uncompressed_bytes=423_902,
        crc32=2_248_400_747,
        sha256="8858cc695976c94943bff1bcb76320bd959b7c771941e6286274f43f5c3f5a0b",
    ),
)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


def fetch_member(member: Member, opener=None) -> bytes:
    """Return verified, bounded TIFF bytes for one fixed ZIP member."""
    if member.end - member.start + 1 != member.compressed_bytes:
        raise ValueError(f"Invalid recorded range for {member.member}")
    if not 0 < member.compressed_bytes <= MAX_COMPRESSED_BYTES:
        raise ValueError(f"Compressed size exceeds cap for {member.member}")
    if not 0 < member.uncompressed_bytes <= MAX_UNCOMPRESSED_BYTES:
        raise ValueError(f"Uncompressed size exceeds cap for {member.member}")

    if opener is None:
        opener = build_opener(_NoRedirect())
    request = Request(
        member.url,
        headers={
            "Range": f"bytes={member.start}-{member.end}",
            "Accept-Encoding": "identity",
            "User-Agent": "szl-microscopy-pilot/0.1 (bounded BBBC006 example)",
        },
    )
    with opener.open(request, timeout=30) as response:
        if response.status != 206:
            raise ValueError(f"Expected HTTP 206 for {member.member}; got {response.status}")
        if response.geturl() != member.url:
            raise ValueError(f"Unexpected source URL for {member.member}")
        content_range = response.headers.get("Content-Range", "")
        match = CONTENT_RANGE_RE.fullmatch(content_range)
        if (
            match is None
            or int(match[1]) != member.start
            or int(match[2]) != member.end
            or int(match[3]) <= member.end
        ):
            raise ValueError(f"Unexpected Content-Range for {member.member}: {content_range!r}")
        if response.headers.get("Content-Length") != str(member.compressed_bytes):
            raise ValueError(f"Unexpected Content-Length for {member.member}")
        if response.headers.get("Content-Encoding", "identity").lower() != "identity":
            raise ValueError(f"Unexpected Content-Encoding for {member.member}")
        compressed = response.read(member.compressed_bytes + 1)
    if len(compressed) != member.compressed_bytes:
        raise ValueError(f"Incomplete or oversized response for {member.member}")

    try:
        inflater = zlib.decompressobj(wbits=-zlib.MAX_WBITS)
        raw = inflater.decompress(compressed, member.uncompressed_bytes + 1)
    except zlib.error as exc:
        raise ValueError(f"Invalid DEFLATE payload for {member.member}") from exc
    if (
        len(raw) != member.uncompressed_bytes
        or not inflater.eof
        or inflater.unused_data
        or inflater.unconsumed_tail
    ):
        raise ValueError(f"Invalid decompressed size or trailing data for {member.member}")
    if zlib.crc32(raw) & 0xFFFFFFFF != member.crc32:
        raise ValueError(f"CRC32 mismatch for {member.member}")
    if hashlib.sha256(raw).hexdigest() != member.sha256:
        raise ValueError(f"SHA-256 mismatch for {member.member}")
    if raw[:4] not in (b"II*\x00", b"MM\x00*"):
        raise ValueError(f"Expected TIFF content for {member.member}")
    return raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_output = Path(__file__).resolve().parents[1] / "examples" / "bbbc006v1" / "images"
    parser.add_argument("--output-dir", type=Path, default=default_output)
    args = parser.parse_args()

    # Verify both members before writing either one.
    downloaded = [(member, fetch_member(member)) for member in MEMBERS]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for member, raw in downloaded:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{member.filename}.", suffix=".tmp", dir=args.output_dir, delete=False
        ) as temporary:
            temporary.write(raw)
            temporary_path = Path(temporary.name)
        try:
            output_path = args.output_dir / member.filename
            os.replace(temporary_path, output_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        print(f"{output_path}: {len(raw)} bytes, SHA-256 {member.sha256}")


if __name__ == "__main__":
    main()
