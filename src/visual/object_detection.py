
import argparse
import json
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


def detect_objects(manifest_path, output_path, model_name, device):
    manifest_path = Path(manifest_path).resolve()
    output_path = Path(output_path).resolve()

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    project_root = Path.cwd()

    # Load the pretrained model once, not once per frame.
    print(f"Loading YOLO model: {model_name}")
    model = YOLO(model_name)

    if device == "0" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but PyTorch cannot access it.")

    detection_dir = output_path.parent / "detections"
    detection_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    print(f"Processing {len(manifest['frames'])} frames...")

    for frame_data in manifest["frames"]:
        frame_path = project_root / frame_data["path"]

        if not frame_path.is_file():
            print(f"Skipping missing frame: {frame_path}")
            continue

        results = model.predict(
            source=str(frame_path),
            device=device,
            conf=0.25,
            verbose=False
        )

        result = results[0]

        frame_objects = []

        if result.boxes is not None:
            boxes = result.boxes

            coordinates = boxes.xyxy.cpu().tolist()
            class_ids = boxes.cls.cpu().tolist()
            confidences = boxes.conf.cpu().tolist()

            for box, class_id, confidence in zip(
                coordinates, class_ids, confidences
            ):
                class_id = int(class_id)

                frame_objects.append({
                    "class_id": class_id,
                    "label": model.names[class_id],
                    "confidence": round(float(confidence), 4),
                    "bounding_box_xyxy": [
                        round(float(value), 2) for value in box
                    ]
                })

        # Save a visual representation of the detections.
        annotated_frame = result.plot()

        annotated_path = (
            detection_dir / f"frame_{frame_data['index']:04d}.jpg"
        )

        if not cv2.imwrite(str(annotated_path), annotated_frame):
            raise RuntimeError(
                f"Could not save annotated frame: {annotated_path}"
            )

        all_results.append({
            "frame_index": frame_data["index"],
            "timestamp_seconds": frame_data["timestamp_seconds"],
            "source_frame": frame_data["path"],
            "annotated_frame": str(
                annotated_path.relative_to(project_root)
            ),
            "objects": frame_objects,
            "object_count": len(frame_objects)
        })

        labels = [obj["label"] for obj in frame_objects]

        print(
            f"[{frame_data['timestamp_seconds']:6.2f}s] "
            f"{len(frame_objects)} objects: {labels}"
        )

    output = {
        "model": model_name,
        "confidence_threshold": 0.25,
        "device": device,
        "total_frames_processed": len(all_results),
        "total_detections": sum(
            frame["object_count"] for frame in all_results
        ),
        "frames": all_results
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(output, indent=4),
        encoding="utf-8"
    )

    print("\nObject detection completed.")
    print(f"Frames processed: {len(all_results)}")
    print(f"Total detections: {output['total_detections']}")
    print(f"JSON output: {output_path}")
    print(f"Annotated frames: {detection_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Run YOLO object detection on extracted video frames."
    )

    parser.add_argument(
        "--manifest",
        default="data/output/ingestion_manifest.json"
    )

    parser.add_argument(
        "--output",
        default="data/output/detections.json"
    )

    parser.add_argument(
        "--model",
        default="yolo11n.pt"
    )

    parser.add_argument(
        "--device",
        default="0",
        help="GPU index (0) or CPU"
    )

    args = parser.parse_args()

    detect_objects(
        args.manifest,
        args.output,
        args.model,
        args.device
    )


if __name__ == "__main__":
    main()
