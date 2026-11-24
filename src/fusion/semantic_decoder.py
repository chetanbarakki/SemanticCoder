
import argparse
import json
import subprocess
import tempfile
import zipfile

from pathlib import Path


def run_command(command):
    subprocess.run(command, check=True)


def decode_package(package_path, output_path, fps=25):
    package_path = Path(package_path).resolve()
    output_path = Path(output_path).resolve()

    if not package_path.exists():
        raise FileNotFoundError(package_path)

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)

        # Extract the semantic package.
        with zipfile.ZipFile(package_path, "r") as archive:
            archive.extractall(root)

        with open(
            root / "semantic_representation.json",
            encoding="utf-8"
        ) as file:
            representation = json.load(file)

        scenes = representation["scenes"]

        duration = max(
            scene["timeline"]["end_seconds"]
            for scene in scenes
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with tempfile.TemporaryDirectory() as work_dir:
            work_dir = Path(work_dir)

            frame_dir = work_dir / "frames"
            audio_dir = work_dir / "audio"

            frame_dir.mkdir()
            audio_dir.mkdir()

            # --------------------------------
            # 1. Reconstruct visual frames
            # --------------------------------

            frame_number = 0

            for scene in scenes:
                scene_duration = (
                    scene["timeline"]["duration_seconds"]
                )

                image_name = Path(
                    scene["keyframe"]["path"]
                ).name

                image_path = root / "keyframes" / image_name

                if not image_path.exists():
                    raise FileNotFoundError(image_path)

                frame_count = round(scene_duration * fps)

                for _ in range(frame_count):
                    destination = (
                        frame_dir /
                        f"frame_{frame_number:06d}.jpg"
                    )

                    destination.write_bytes(
                        image_path.read_bytes()
                    )

                    frame_number += 1

            # --------------------------------
            # 2. Generate speech using eSpeak
            # --------------------------------

            speech_files = []

            for scene in scenes:
                segments = scene[
                    "audio_observations"
                ]["speech_segments"]

                for index, segment in enumerate(segments):
                    text = segment["text"]
                    start = segment["start_seconds"]

                    audio_path = (
                        audio_dir /
                        f"speech_{len(speech_files):03d}.wav"
                    )

                    run_command([
                        "espeak-ng",
                        "-v", "en-us",
                        "-s", "150",
                        "-w", str(audio_path),
                        text
                    ])

                    speech_files.append({
                        "path": audio_path,
                        "start": start
                    })

            # --------------------------------
            # 3. Synchronize speech segments
            # --------------------------------

            ffmpeg_command = [
                "ffmpeg",
                "-y",
                "-framerate", str(fps),
                "-i", str(frame_dir / "frame_%06d.jpg")
            ]

            for speech in speech_files:
                ffmpeg_command.extend([
                    "-i", str(speech["path"])
                ])

            filters = []

            for index, speech in enumerate(speech_files):
                delay_ms = round(speech["start"] * 1000)

                filters.append(
                    f"[{index + 1}:a]"
                    f"adelay={delay_ms}|{delay_ms}"
                    f"[a{index}]"
                )

            audio_labels = "".join(
                f"[a{i}]"
                for i in range(len(speech_files))
            )

            if speech_files:
                filters.append(
                    f"{audio_labels}"
                    f"amix=inputs={len(speech_files)}:"
                    f"duration=longest:normalize=0,"
                    f"apad,atrim=duration={duration}"
                    f"[aout]"
                )

                ffmpeg_command.extend([
                    "-filter_complex",
                    ";".join(filters),
                    "-map", "0:v",
                    "-map", "[aout]"
                ])

            else:
                ffmpeg_command.extend([
                    "-map", "0:v"
                ])

            ffmpeg_command.extend([
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p"
            ])

            if speech_files:
                ffmpeg_command.extend([
                    "-c:a", "aac",
                    "-b:a", "128k"
                ])

            ffmpeg_command.extend([
                "-t", str(duration),
                "-movflags", "+faststart",
                str(output_path)
            ])

            run_command(ffmpeg_command)

        print("\nSemantic decoding completed.")
        print(f"Scenes reconstructed: {len(scenes)}")
        print(f"Frames generated: {frame_number}")
        print(f"Speech segments generated: {len(speech_files)}")
        print(f"Output video: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Decode a semantic video package."
    )

    parser.add_argument(
        "--package",
        default="data/output/semantic_package.zip"
    )

    parser.add_argument(
        "--output",
        default="data/output/reconstructed_video.mp4"
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=25
    )

    args = parser.parse_args()

    decode_package(
        args.package,
        args.output,
        args.fps
    )


if __name__ == "__main__":
    main()
