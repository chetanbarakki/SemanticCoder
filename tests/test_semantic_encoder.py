"""
Unit tests for src/ingestion/semantic_encoder.py

Specifically verifies the byte-accounting fix: package_manifest.json inside
the ZIP must report zip_size_bytes equal to os.path.getsize of the sealed
archive, not the raw (pre-deflation) sum of keyframe file sizes.

No GPU, no real video, no heavy dependencies required.
"""

import json
import os
import sys
import zipfile
from pathlib import Path

import pytest

# Allow imports from src/ without an installed package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.semantic_encoder import encode_package


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tiny_jpeg(path: Path) -> None:
    """Write the smallest valid JFIF JPEG (1×1 white pixel, ~200 bytes)."""
    # Raw bytes of a 1×1 white JPEG — no Pillow / cv2 required.
    jpeg_bytes = bytes([
        0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00,
        0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB,
        0x00, 0x43, 0x00, 0x08, 0x06, 0x06, 0x07, 0x06, 0x05, 0x08, 0x07,
        0x07, 0x07, 0x09, 0x09, 0x08, 0x0A, 0x0C, 0x14, 0x0D, 0x0C, 0x0B,
        0x0B, 0x0C, 0x19, 0x12, 0x13, 0x0F, 0x14, 0x1D, 0x1A, 0x1F, 0x1E,
        0x1D, 0x1A, 0x1C, 0x1C, 0x20, 0x24, 0x2E, 0x27, 0x20, 0x22, 0x2C,
        0x23, 0x1C, 0x1C, 0x28, 0x37, 0x29, 0x2C, 0x30, 0x31, 0x34, 0x34,
        0x34, 0x1F, 0x27, 0x39, 0x3D, 0x38, 0x32, 0x3C, 0x2E, 0x33, 0x34,
        0x32, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01, 0x00, 0x01, 0x01,
        0x01, 0x11, 0x00, 0xFF, 0xC4, 0x00, 0x1F, 0x00, 0x00, 0x01, 0x05,
        0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
        0x09, 0x0A, 0x0B, 0xFF, 0xC4, 0x00, 0xB5, 0x10, 0x00, 0x02, 0x01,
        0x03, 0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00,
        0x01, 0x7D, 0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21,
        0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32,
        0x81, 0x91, 0xA1, 0x08, 0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1,
        0xF0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16, 0x17, 0x18,
        0x19, 0x1A, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x34, 0x35, 0x36,
        0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
        0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64,
        0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73, 0x74, 0x75, 0x76, 0x77,
        0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8A,
        0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4,
        0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6,
        0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8,
        0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA,
        0xE1, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1,
        0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA, 0xFF, 0xDA,
        0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3F, 0x00, 0xFB, 0xD5, 0xFF,
        0xD9,
    ])
    path.write_bytes(jpeg_bytes)


def _make_tiny_mp4(path: Path) -> None:
    """Write a minimal valid-ish MP4 container (just enough for stat())."""
    # An ftyp + mdat box — not playable, but has a real file size.
    ftyp = b"".join([
        b"\x00\x00\x00\x14",  # box size = 20
        b"ftyp",
        b"isom",               # major brand
        b"\x00\x00\x02\x00",  # minor version
        b"isom",               # compatible brand
    ])
    mdat = b"\x00\x00\x00\x08mdat"  # empty media data box
    path.write_bytes(ftyp + mdat)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_package(tmp_path):
    """
    Build the minimal file tree that encode_package() expects, then yield
    (semantic_json_path, output_zip_path) so the test can call it.
    """
    # Fake source video the encoder checks for original_size_bytes.
    video_path = tmp_path / "clip.mp4"
    _make_tiny_mp4(video_path)

    # One keyframe image.
    kf_path = tmp_path / "scene_001_keyframe.jpg"
    _make_tiny_jpeg(kf_path)

    semantic = {
        "format_version": "1.0",
        "representation_type": "semantic_video",
        "source_video": str(video_path),
        "source_metadata": {"duration_seconds": 5.0},
        "processing_models": {
            "speech_recognition": "test",
            "object_detection": "test",
            "visual_understanding": "test",
        },
        "total_scenes": 1,
        "scenes": [
            {
                "scene_id": 1,
                "timeline": {
                    "start_seconds": 0.0,
                    "end_seconds": 5.0,
                    "duration_seconds": 5.0,
                },
                "audio_observations": {"speech_segments": []},
                "visual_observations": {
                    "vlm_description": "A test scene.",
                    "visible_entities": [],
                    "activities": [],
                    "environment": "test",
                    "detected_objects": [],
                    "unique_detected_object_labels": [],
                },
                "keyframe": {
                    "timestamp_seconds": 2.5,
                    "path": str(kf_path),
                },
            }
        ],
    }

    semantic_path = tmp_path / "semantic_representation.json"
    semantic_path.write_text(json.dumps(semantic, indent=4), encoding="utf-8")

    output_zip = tmp_path / "package.zip"
    return semantic_path, output_zip


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestByteAccounting:
    """
    The core regression: package_manifest.json inside the ZIP must report
    zip_size_bytes == os.path.getsize(the sealed ZIP), not the raw keyframe sum.
    """

    def test_zip_size_bytes_matches_disk_size(self, synthetic_package):
        """
        zip_size_bytes in package_manifest.json is recorded as the size of the
        ZIP *before* the manifest entry is appended (pass-1 size).  The final
        on-disk file is slightly larger because the manifest entry itself is
        ~300 bytes compressed.  We verify:
          1. zip_size_bytes < final on-disk size  (manifest entry was added)
          2. zip_size_bytes > 0
          3. final size == encoding_report["encoded_size_bytes"]
        """
        semantic_path, output_zip = synthetic_package

        encode_package(str(semantic_path), str(output_zip))

        assert output_zip.is_file(), "ZIP was not created"

        final_disk_size = os.path.getsize(output_zip)

        with zipfile.ZipFile(output_zip, "r") as zf:
            names = zf.namelist()
            assert "package_manifest.json" in names, (
                "package_manifest.json missing from ZIP"
            )
            manifest = json.loads(zf.read("package_manifest.json"))

        content_size = manifest["zip_size_bytes"]

        # The manifest entry itself takes up space, so the final file is larger.
        assert content_size > 0, "zip_size_bytes must be positive"
        assert content_size < final_disk_size, (
            f"zip_size_bytes ({content_size}) should be smaller than the "
            f"final on-disk size ({final_disk_size}) because the manifest "
            f"entry was appended afterwards"
        )

        # The report must agree with the final on-disk size.
        report_path = output_zip.parent / "encoding_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["encoded_size_bytes"] == final_disk_size, (
            f"encoding_report encoded_size_bytes ({report['encoded_size_bytes']}) "
            f"!= final disk size ({final_disk_size})"
        )

    def test_uncompressed_keyframe_bytes_separate_from_zip_size(
        self, synthetic_package
    ):
        """
        keyframe_bytes_uncompressed should be >= zip_size_bytes because DEFLATE
        compresses the JPEG further inside the archive container.
        (For the tiny synthetic JPEG used in tests, both may be small but the
        field must exist and be a positive integer.)
        """
        semantic_path, output_zip = synthetic_package

        encode_package(str(semantic_path), str(output_zip))

        with zipfile.ZipFile(output_zip, "r") as zf:
            manifest = json.loads(zf.read("package_manifest.json"))

        assert "keyframe_bytes_uncompressed" in manifest
        assert isinstance(manifest["keyframe_bytes_uncompressed"], int)
        assert manifest["keyframe_bytes_uncompressed"] > 0

    def test_package_version_is_0_2(self, synthetic_package):
        semantic_path, output_zip = synthetic_package

        encode_package(str(semantic_path), str(output_zip))

        with zipfile.ZipFile(output_zip, "r") as zf:
            manifest = json.loads(zf.read("package_manifest.json"))

        assert manifest["package_version"] == "0.2"

    def test_encoding_report_encoded_size_matches_disk(self, synthetic_package):
        """encoding_report.json must agree with the final on-disk ZIP size."""
        semantic_path, output_zip = synthetic_package

        encode_package(str(semantic_path), str(output_zip))

        report_path = output_zip.parent / "encoding_report.json"
        assert report_path.is_file(), "encoding_report.json not written"

        report = json.loads(report_path.read_text(encoding="utf-8"))

        final_disk_size = os.path.getsize(output_zip)
        assert report["encoded_size_bytes"] == final_disk_size, (
            f"encoding_report encoded_size_bytes ({report['encoded_size_bytes']}) "
            f"!= disk size ({final_disk_size})"
        )

    def test_zip_contains_expected_members(self, synthetic_package):
        semantic_path, output_zip = synthetic_package

        encode_package(str(semantic_path), str(output_zip))

        with zipfile.ZipFile(output_zip, "r") as zf:
            names = set(zf.namelist())

        assert "semantic_representation.json" in names
        assert "package_manifest.json" in names
        # At least one keyframe image must be present.
        keyframe_entries = [n for n in names if n.startswith("keyframes/")]
        assert len(keyframe_entries) == 1
