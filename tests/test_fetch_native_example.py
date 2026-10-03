"""Protocol and integrity checks for the bounded native-image fetcher."""

from dataclasses import replace
import hashlib
import io
from pathlib import Path
import sys
import unittest
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from fetch_native_example import Member, fetch_member  # noqa: E402


class Response(io.BytesIO):
    def __init__(self, payload, member, *, status=206, content_range=None, content_length=None):
        super().__init__(payload)
        self.status = status
        self.headers = {
            "Content-Range": content_range
            or f"bytes {member.start}-{member.end}/{member.end + 100}",
            "Content-Length": str(member.compressed_bytes)
            if content_length is None
            else content_length,
        }
        self.url = member.url

    def geturl(self):
        return self.url


class Opener:
    def __init__(self, response):
        self.response = response

    def open(self, request, timeout):
        assert request.get_header("Range") == f"bytes={self.response.member.start}-{self.response.member.end}"
        assert timeout == 30
        return self.response


class FetchTests(unittest.TestCase):
    def setUp(self):
        self.raw = b"II*\x00" + bytes(range(256)) * 2
        compressor = zlib.compressobj(wbits=-zlib.MAX_WBITS)
        self.compressed = compressor.compress(self.raw) + compressor.flush()
        self.member = Member(
            url="https://data.broadinstitute.org/example.zip",
            member="test/example.tif",
            start=100,
            end=100 + len(self.compressed) - 1,
            compressed_bytes=len(self.compressed),
            uncompressed_bytes=len(self.raw),
            crc32=zlib.crc32(self.raw) & 0xFFFFFFFF,
            sha256=hashlib.sha256(self.raw).hexdigest(),
        )

    def fetch(self, payload=None, member=None, **response_kwargs):
        member = member or self.member
        response = Response(
            self.compressed if payload is None else payload,
            member,
            **response_kwargs,
        )
        response.member = member
        return fetch_member(member, Opener(response))

    def test_valid_member(self):
        self.assertEqual(self.fetch(), self.raw)

    def test_rejects_whole_archive_response(self):
        with self.assertRaisesRegex(ValueError, "HTTP 206"):
            self.fetch(status=200)

    def test_rejects_wrong_content_range(self):
        with self.assertRaisesRegex(ValueError, "Content-Range"):
            self.fetch(content_range="bytes 0-10/1000")

    def test_rejects_short_payload(self):
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            self.fetch(payload=self.compressed[:-1])

    def test_rejects_wrong_hash(self):
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.fetch(member=replace(self.member, sha256="0" * 64))

    def test_rejects_inflation_past_expected_size(self):
        with self.assertRaisesRegex(ValueError, "decompressed size"):
            self.fetch(member=replace(self.member, uncompressed_bytes=len(self.raw) - 1))


if __name__ == "__main__":
    unittest.main()
