
import argparse
import json
import shutil
from pathlib import Path


def load_json(path):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def select_midpoint_frame(scene, frames):
    midpoint = (
        scene["start_seconds"] + scene["end_seconds"]
    ) / 2

    # Find the extracted frame closest to the scene midpoint.
    return min(
        frames,
        key=lambda frame: abs(
            frame["timestamp_seconds"] - midpoint
        )
    )


def extract_keyframes(manifest_path, scenes_path, output_path):
    manifest = load_json(manifest_path)
    scene_data = load_json(scenes_path)

    frames = manifest["frames"]
    keyframe_dir = Path(output_path)
    keyframe_dir.mkdir(parents=True, exist_ok=True)

    selected_keyframes = []

    for scene in scene_data["scenes"]:
        selected = select_midpoint_frame(scene, frames)

        source_path = Path(selected["path"])

        if not source_path.is_file():
            raise FileNotFoundError(
                f"Frame not found: {source_path}"
            )

        destination = (
            keyframe_dir /
            f"scene_{scene['scene_id']:03d}_keyframe.jpg"
        )

        shutil.copy2(source_path, destination)

        selected_keyframes.append({
            "scene_id": scene["scene_id"],
            "scene_start_seconds": scene["start_seconds"],
            "scene_end_seconds": scene["end_seconds"],
            "timestamp_seconds": selected["timestamp_seconds"],
            "original_frame_path": str(source_path),
            "keyframe_path": str(destination)
        })

        print(
            f"Scene {scene['scene_id']}: "
            f"selected frame at "
            f"{selected['timestamp_seconds']:.2f}s"
        )

    output = {
        "source_video": manifest["source_video"],
        "selection_method": "scene_midpoint",
        "total_keyframes": len(selected_keyframes),
        "keyframes": selected_keyframes
    }

    output_json = Path("data/output/keyframes.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)

    output_json.write_text(
        json.dumps(output, indent=4),
        encoding="utf-8"
    )

    print("\nKeyframe extraction completed.")
    print(f"Keyframes selected: {len(selected_keyframes)}")
    print(f"Keyframe directory: {keyframe_dir}")
    print(f"JSON output: {output_json}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract representative frames for each scene."
    )

    parser.add_argument(
        "--manifest",
        default="data/output/ingestion_manifest.json"
    )

    parser.add_argument(
        "--scenes",
        default="data/output/scenes.json"
    )

    parser.add_argument(
        "--output",
        default="data/keyframes"
    )

    args = parser.parse_args()

    extract_keyframes(
        args.manifest,
        args.scenes,
        args.output
    )


if __name__ == "__main__":
    main()
