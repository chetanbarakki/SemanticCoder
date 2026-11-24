
import argparse
import json
from pathlib import Path

from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector


def detect_scenes(video_path, output_path, threshold=27.0):
    video_path = Path(video_path).resolve()
    output_path = Path(output_path).resolve()

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    print(f"Opening video: {video_path}")

    video = open_video(str(video_path))

    scene_manager = SceneManager()

    # Detect significant visual changes.
    scene_manager.add_detector(
        ContentDetector(
            threshold=threshold,
            min_scene_len=15
        )
    )

    print("Detecting scene boundaries...")

    scene_manager.detect_scenes(video=video)
    scene_list = scene_manager.get_scene_list()

    scenes = []

    for index, (start, end) in enumerate(scene_list):
        scene = {
            "scene_id": index + 1,
            "start_seconds": round(start.get_seconds(), 3),
            "end_seconds": round(end.get_seconds(), 3),
            "duration_seconds": round(
                end.get_seconds() - start.get_seconds(), 3
            ),
            "start_frame": start.get_frames(),
            "end_frame": end.get_frames()
        }

        scenes.append(scene)

    result = {
        "source_video": str(video_path),
        "detector": "ContentDetector",
        "threshold": threshold,
        "total_scenes": len(scenes),
        "scenes": scenes
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(result, indent=4),
        encoding="utf-8"
    )

    print("\nScene detection completed.")
    print(f"Total scenes detected: {len(scenes)}")

    for scene in scenes:
        print(
            f"Scene {scene['scene_id']}: "
            f"{scene['start_seconds']:.2f}s - "
            f"{scene['end_seconds']:.2f}s"
        )

    print(f"\nOutput: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Detect visual scene changes in a video."
    )

    parser.add_argument(
        "--input",
        default="data/input/test_video.mp4"
    )

    parser.add_argument(
        "--output",
        default="data/output/scenes.json"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=27.0
    )

    args = parser.parse_args()

    detect_scenes(
        args.input,
        args.output,
        args.threshold
    )


if __name__ == "__main__":
    main()
