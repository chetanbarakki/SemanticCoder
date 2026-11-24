
import argparse
import json
from pathlib import Path


def load_json(path):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Required input not found: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def find_overlapping_segments(start, end, segments):
    """Return speech segments overlapping a scene's time interval."""
    overlapping = []

    for segment in segments:
        segment_start = segment["start"]
        segment_end = segment["end"]

        overlap = min(end, segment_end) - max(start, segment_start)

        if overlap >= 0.05:
            overlapping.append({
                "start_seconds": segment_start,
                "end_seconds": segment_end,
                "text": segment["text"],
                "overlap_seconds": round(overlap, 3)
            })

    return overlapping


def find_frame_detections(start, end, frames):
    """Collect object detections from frames inside a scene."""
    detections = []

    for frame in frames:
        timestamp = frame["timestamp_seconds"]

        if start <= timestamp < end:
            for obj in frame.get("objects", []):
                detections.append({
                    "timestamp_seconds": timestamp,
                    "label": obj["label"],
                    "confidence": obj["confidence"],
                    "bounding_box_xyxy": obj["bounding_box_xyxy"]
                })

    return detections


def fuse_timelines(
    transcription_path,
    scenes_path,
    detections_path,
    output_path
):
    transcription = load_json(transcription_path)
    scene_data = load_json(scenes_path)
    detection_data = load_json(detections_path)

    fused_scenes = []

    for scene in scene_data["scenes"]:
        start = scene["start_seconds"]
        end = scene["end_seconds"]

        speech = find_overlapping_segments(
            start,
            end,
            transcription["segments"]
        )

        objects = find_frame_detections(
            start,
            end,
            detection_data["frames"]
        )

        fused_scenes.append({
            "scene_id": scene["scene_id"],
            "start_seconds": start,
            "end_seconds": end,
            "duration_seconds": scene["duration_seconds"],
            "speech": speech,
            "visual_observations": {
                "objects": objects,
                "unique_object_labels": sorted({
                    obj["label"] for obj in objects
                })
            }
        })

    output = {
        "source_video": scene_data["source_video"],
        "total_scenes": len(fused_scenes),
        "fusion_method": "timestamp_overlap",
        "scenes": fused_scenes
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(output, indent=4),
        encoding="utf-8"
    )

    print("\nTemporal fusion completed.")
    print(f"Scenes fused: {len(fused_scenes)}")

    for scene in fused_scenes:
        print(
            f"\nScene {scene['scene_id']}: "
            f"{scene['start_seconds']:.2f}s - "
            f"{scene['end_seconds']:.2f}s"
        )

        for speech in scene["speech"]:
            print(f"  Speech: {speech['text']}")

        print(
            "  Detected objects:",
            scene["visual_observations"]["unique_object_labels"]
        )

    print(f"\nOutput: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Fuse audio transcription and visual observations."
    )

    parser.add_argument(
        "--transcription",
        default="data/output/transcription.json"
    )

    parser.add_argument(
        "--scenes",
        default="data/output/scenes.json"
    )

    parser.add_argument(
        "--detections",
        default="data/output/detections.json"
    )

    parser.add_argument(
        "--output",
        default="data/output/fused_timeline.json"
    )

    args = parser.parse_args()

    fuse_timelines(
        args.transcription,
        args.scenes,
        args.detections,
        args.output
    )


if __name__ == "__main__":
    main()
