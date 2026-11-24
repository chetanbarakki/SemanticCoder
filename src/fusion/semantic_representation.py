
import argparse
import json
from pathlib import Path


def load_json(path):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def get_speech_for_scene(start, end, segments):
    observations = []

    for segment in segments:
        segment_start = segment["start"]
        segment_end = segment["end"]

        overlap = min(end, segment_end) - max(start, segment_start)

        # Ignore negligible overlaps at scene boundaries.
        if overlap >= 0.05:
            observations.append({
                "start_seconds": segment_start,
                "end_seconds": segment_end,
                "text": segment["text"],
                "overlap_seconds": round(overlap, 3)
            })

    return observations


def get_objects_for_scene(start, end, frames):
    observations = []

    for frame in frames:
        timestamp = frame["timestamp_seconds"]

        if start <= timestamp < end:
            for obj in frame.get("objects", []):
                observations.append({
                    "timestamp_seconds": timestamp,
                    "label": obj["label"],
                    "confidence": obj["confidence"],
                    "bounding_box_xyxy": obj["bounding_box_xyxy"]
                })

    return observations


def build_semantic_representation(
    scenes_path,
    transcription_path,
    detections_path,
    understanding_path,
    keyframes_path,
    output_path
):
    scenes_data = load_json(scenes_path)
    transcription_data = load_json(transcription_path)
    detections_data = load_json(detections_path)
    understanding_data = load_json(understanding_path)
    keyframes_data = load_json(keyframes_path)

    # Index model outputs by scene ID for reliable matching.
    understanding_by_id = {
        item["scene_id"]: item
        for item in understanding_data["scenes"]
    }

    keyframes_by_id = {
        item["scene_id"]: item
        for item in keyframes_data["keyframes"]
    }

    semantic_scenes = []

    for scene in scenes_data["scenes"]:
        scene_id = scene["scene_id"]
        start = scene["start_seconds"]
        end = scene["end_seconds"]

        if scene_id not in understanding_by_id:
            raise ValueError(
                f"Missing VLM description for scene {scene_id}"
            )

        if scene_id not in keyframes_by_id:
            raise ValueError(
                f"Missing keyframe for scene {scene_id}"
            )

        visual = understanding_by_id[scene_id]
        keyframe = keyframes_by_id[scene_id]

        speech = get_speech_for_scene(
            start,
            end,
            transcription_data["segments"]
        )

        objects = get_objects_for_scene(
            start,
            end,
            detections_data["frames"]
        )

        semantic_scenes.append({
            "scene_id": scene_id,

            "timeline": {
                "start_seconds": start,
                "end_seconds": end,
                "duration_seconds": scene["duration_seconds"]
            },

            "audio_observations": {
                "speech_segments": speech
            },

            "visual_observations": {
                "vlm_description": visual["description"],
                "visible_entities": visual["visible_entities"],
                "activities": visual["activities"],
                "environment": visual["environment"],

                "detected_objects": objects,

                "unique_detected_object_labels": sorted({
                    obj["label"] for obj in objects
                })
            },

            "keyframe": {
                "timestamp_seconds": keyframe["timestamp_seconds"],
                "path": keyframe["keyframe_path"]
            }
        })

    output = {
        "format_version": "1.0",
        "representation_type": "semantic_video",
        "source_video": scenes_data["source_video"],

        "source_metadata": {
            "duration_seconds": load_json(
                "data/output/ingestion_manifest.json"
            )["metadata"]["duration_seconds"]
        },

        "processing_models": {
            "speech_recognition": "Faster-Whisper",
            "object_detection": "YOLO",
            "visual_understanding": understanding_data["model"]
        },

        "total_scenes": len(semantic_scenes),
        "scenes": semantic_scenes
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(output, indent=4),
        encoding="utf-8"
    )

    print("\nSemantic representation created.")
    print(f"Scenes represented: {len(semantic_scenes)}")
    print(f"Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build a unified semantic video representation."
    )

    parser.add_argument(
        "--scenes",
        default="data/output/scenes.json"
    )

    parser.add_argument(
        "--transcription",
        default="data/output/transcription.json"
    )

    parser.add_argument(
        "--detections",
        default="data/output/detections.json"
    )

    parser.add_argument(
        "--understanding",
        default="data/output/scene_understanding.json"
    )

    parser.add_argument(
        "--keyframes",
        default="data/output/keyframes.json"
    )

    parser.add_argument(
        "--output",
        default="data/output/semantic_representation.json"
    )

    args = parser.parse_args()

    build_semantic_representation(
        args.scenes,
        args.transcription,
        args.detections,
        args.understanding,
        args.keyframes,
        args.output
    )


if __name__ == "__main__":
    main()
