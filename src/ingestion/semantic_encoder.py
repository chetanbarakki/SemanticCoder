"""
Semantic Encoder — packages a semantic video representation for transport.

Inputs:
    semantic_representation.json  – fused scene data produced by
                                    src/fusion/semantic_representation.py
    keyframe image files          – paths referenced inside the JSON

Outputs:
    semantic_package.zip          – self-contained ZIP with the JSON,
                                    all keyframe images, and a manifest
    encoding_report.json          – byte-size comparison (original video
                                    vs. final ZIP, using os.path.getsize
                                    on the sealed archive)

Failure mode:
    If a keyframe path listed in the semantic JSON does not exist on disk
    the encoder raises FileNotFoundError before creating any output.
"""

import argparse
import json
import os
import shutil
import zipfile
from pathlib import Path


def load_json(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def encode_package(semantic_path, output_path):
    semantic_path = Path(semantic_path)
    output_path = Path(output_path)

    semantic = load_json(semantic_path)

    # Use a temporary directory to assemble a portable package.
    staging = output_path.parent / "semantic_package_staging"

    if staging.exists():
        shutil.rmtree(staging)

    keyframe_dir = staging / "keyframes"
    keyframe_dir.mkdir(parents=True, exist_ok=True)

    packaged_semantic = semantic.copy()
    packaged_semantic["source_video"] = Path(
        semantic["source_video"]
    ).name

    packaged_scenes = []

    total_keyframe_bytes = 0

    for scene in semantic["scenes"]:
        scene_copy = scene.copy()
        keyframe = scene["keyframe"]

        original_path = Path(keyframe["path"])

        if not original_path.is_file():
            raise FileNotFoundError(
                f"Keyframe missing for scene "
                f"{scene['scene_id']}: {original_path}"
            )

        filename = original_path.name
        destination = keyframe_dir / filename

        shutil.copy2(original_path, destination)

        total_keyframe_bytes += destination.stat().st_size

        # Store a relative path so the package works elsewhere.
        scene_copy["keyframe"] = {
            "timestamp_seconds": keyframe["timestamp_seconds"],
            "path": f"keyframes/{filename}"
        }

        packaged_scenes.append(scene_copy)

    packaged_semantic["scenes"] = packaged_scenes

    semantic_output = staging / "semantic_representation.json"
    semantic_output.write_text(
        json.dumps(packaged_semantic, indent=4),
        encoding="utf-8"
    )

    # ----------------------------------------------------------------
    # Byte-accounting strategy (two-pass):
    #
    # Pass 1 — zip all content files (no manifest yet), measure the
    #           sealed size with os.path.getsize.  That size is the
    #           "content-only" baseline.
    #
    # Pass 2 — append package_manifest.json (which now contains the
    #           correct zip_size_bytes from pass 1) and re-measure.
    #           The final on-disk size is what goes into encoding_report.
    #
    # Why two passes instead of writing the manifest to the staging dir
    # and zipping everything at once?  If the manifest were staged and
    # zipped in one go, zip_size_bytes could only be known *after* the
    # ZIP is sealed, creating a chicken-and-egg problem.  The two-pass
    # approach resolves that: pass-1 gives us the content size, pass-2
    # stamps it into the manifest and appends it.  The difference between
    # the two on-disk measurements is just the compressed manifest entry
    # (~300 bytes) — negligible but precisely accounted for.
    # ----------------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        output_path.unlink()

    # Pass 1: content files only (no manifest).
    with zipfile.ZipFile(
        output_path,
        "w",
        compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for file_path in staging.rglob("*"):
            if file_path.is_file():
                archive.write(
                    file_path,
                    file_path.relative_to(staging)
                )

    # Size of the content-only ZIP — used as zip_size_bytes in the manifest.
    content_zip_bytes = os.path.getsize(output_path)

    # Pass 2: append the manifest that now carries the correct size stamp.
    manifest = {
        "package_format": "semantic-codec-prototype",
        "package_version": "0.2",
        "contents": [
            "semantic_representation.json",
            "keyframes/",
            "package_manifest.json",
        ],
        "total_scenes": len(packaged_scenes),
        "total_keyframes": len(packaged_scenes),
        # Raw (uncompressed) sum of keyframe file sizes — useful for
        # understanding visual data volume before compression.
        "keyframe_bytes_uncompressed": total_keyframe_bytes,
        # Size of the ZIP *before* this manifest entry was appended.
        # A decoder can re-derive the manifest's own compressed size as
        # os.path.getsize(zip) - zip_size_bytes if needed.
        "zip_size_bytes": content_zip_bytes,
    }

    with zipfile.ZipFile(
        output_path,
        "a",
        compression=zipfile.ZIP_DEFLATED
    ) as archive:
        archive.writestr(
            "package_manifest.json",
            json.dumps(manifest, indent=4)
        )

    # Final on-disk size (content + manifest entry) for the report.
    final_zip_bytes = os.path.getsize(output_path)

    # Measure the original and encoded sizes.
    original_video = Path(semantic["source_video"])

    if not original_video.is_file():
        raise FileNotFoundError(
            f"Original video not found: {original_video}"
        )

    original_bytes = original_video.stat().st_size

    report = {
        "original_video": original_video.name,
        "original_size_bytes": original_bytes,
        "encoded_package": output_path.name,
        "encoded_size_bytes": final_zip_bytes,
        "keyframe_data_bytes_uncompressed": total_keyframe_bytes,
        "size_reduction_percent": round(
            (1 - final_zip_bytes / original_bytes) * 100, 2
        ),
        "size_ratio_original_to_package": round(
            original_bytes / final_zip_bytes, 2
        ) if final_zip_bytes else None,
        "note": (
            "Semantic, lossy representation. "
            "Not a quality-equivalent codec comparison."
        )
    }

    report_path = output_path.parent / "encoding_report.json"
    report_path.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8"
    )

    shutil.rmtree(staging)

    print("\nSemantic encoding completed.")
    print(f"Package: {output_path}")
    print(f"Original size: {original_bytes:,} bytes")
    print(f"Package size:  {final_zip_bytes:,} bytes")
    print(
        f"Size reduction: "
        f"{report['size_reduction_percent']}%"
    )
    print(f"Report: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Package semantic video representation."
    )

    parser.add_argument(
        "--input",
        default="data/output/semantic_representation.json"
    )

    parser.add_argument(
        "--output",
        default="data/output/semantic_package.zip"
    )

    args = parser.parse_args()

    encode_package(args.input, args.output)


if __name__ == "__main__":
    main()
