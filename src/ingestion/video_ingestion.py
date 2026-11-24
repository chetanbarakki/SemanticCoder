
import argparse
import json
import shutil
import subprocess
from pathlib import Path

import cv2


def run_command(command):
    """Run a system command and raise a useful error if it fails."""
    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())

    return result.stdout


def get_video_metadata(video_path):
    """Read video and audio stream information using FFprobe."""
    output = run_command([
        "ffprobe",
        "-v", "error",
        "-show_streams",
        "-show_format",
        "-of", "json",
        str(video_path)
    ])

    data = json.loads(output)

    video_stream = next(
        (s for s in data["streams"] if s["codec_type"] == "video"),
        None
    )
    audio_stream = next(
        (s for s in data["streams"] if s["codec_type"] == "audio"),
        None
    )

    if video_stream is None:
        raise ValueError("Input file does not contain a video stream.")

    duration = float(
        video_stream.get("duration")
        or data["format"].get("duration")
        or 0
    )

    return {
        "duration_seconds": duration,
        "width": video_stream.get("width"),
        "height": video_stream.get("height"),
        "fps": video_stream.get("r_frame_rate"),
        "frame_count": int(video_stream.get("nb_frames", 0) or 0),
        "video_codec": video_stream.get("codec_name"),
        "audio_codec": (
            audio_stream.get("codec_name") if audio_stream else None
        ),
        "has_audio": audio_stream is not None
    }


def extract_audio(video_path, audio_path):
    """Extract mono 16 kHz PCM audio for speech recognition."""
    audio_path.parent.mkdir(parents=True, exist_ok=True)

    run_command([
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(audio_path)
    ])

    return {
        "path": str(audio_path),
        "sample_rate": 16000,
        "channels": 1,
        "format": "wav"
    }


def extract_frames(video_path, frames_dir, duration, sample_fps):
    """Extract timestamped frames at a configurable sampling rate."""
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Remove only old JPEGs from this video's dedicated output folder.
    for old_frame in frames_dir.glob("frame_*.jpg"):
        old_frame.unlink()

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    interval = 1.0 / sample_fps
    frames = []
    timestamp = 0.0
    index = 0

    while timestamp < duration:
        capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        success, frame = capture.read()

        if not success:
            break

        filename = f"frame_{index:04d}.jpg"
        frame_path = frames_dir / filename

        written = cv2.imwrite(
            str(frame_path),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 90]
        )

        if not written:
            capture.release()
            raise RuntimeError(f"Could not write frame: {frame_path}")

        frames.append({
            "index": index,
            "timestamp_seconds": round(timestamp, 3),
            "path": str(frame_path)
        })

        index += 1
        timestamp = index * interval

    capture.release()
    return frames


def ingest_video(video_path, output_root, sample_fps):
    """Run the complete ingestion pipeline.

    output_root is the directory that will receive ingestion_manifest.json
    directly (e.g. data/output or data/output_real).  Audio and frames are
    written one level up from output_root, under audio/ and frames/, so they
    are shared across different output configurations for the same clip.

    Previously output_root was the *parent* of the output directory and the
    code appended "output" internally.  That made it impossible for callers
    to redirect outputs to a differently-named directory (e.g. output_real)
    without also breaking the audio/frames paths.  The fix passes output_root
    as the final destination directory and derives sibling directories from
    output_root.parent.
    """
    video_path = video_path.resolve()
    output_root = output_root.resolve()

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if sample_fps <= 0:
        raise ValueError("sample_fps must be greater than zero.")

    metadata = get_video_metadata(video_path)

    if metadata["duration_seconds"] <= 0:
        raise ValueError("Could not determine video duration.")

    video_name = video_path.stem

    # Audio and frames live under data/ (sibling of output_root) so they are
    # clip-specific but not duplicated per output configuration.
    data_root = output_root.parent
    audio_path = data_root / "audio" / f"{video_name}.wav"
    frames_dir = data_root / "frames" / video_name

    # The manifest is written directly into output_root so the caller can
    # name that directory anything (output, output_real, etc.).
    manifest_path = output_root / "ingestion_manifest.json"

    audio_info = None
    if metadata["has_audio"]:
        audio_info = extract_audio(video_path, audio_path)

    frames = extract_frames(
        video_path,
        frames_dir,
        metadata["duration_seconds"],
        sample_fps
    )

    # Store paths relative to the project root for portability.
    project_root = Path.cwd()

    if audio_info:
        audio_info["path"] = str(
            Path(audio_info["path"]).resolve().relative_to(project_root)
        )

    for frame in frames:
        frame["path"] = str(
            Path(frame["path"]).resolve().relative_to(project_root)
        )

    manifest = {
        "source_video": str(video_path.relative_to(project_root)),
        "metadata": metadata,
        "audio": audio_info,
        "sampling": {
            "frames_per_second": sample_fps,
            "total_frames_extracted": len(frames)
        },
        "frames": frames
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=4),
        encoding="utf-8"
    )

    print("\nIngestion completed successfully.")
    print(f"Duration: {metadata['duration_seconds']:.2f} seconds")
    print(f"Audio extracted: {audio_info is not None}")
    print(f"Frames extracted: {len(frames)}")
    print(f"Manifest: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract audio, frames and metadata from a video."
    )
    parser.add_argument(
        "--input",
        default="data/input/test_video.mp4",
        help="Path to the input video"
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=1.0,
        help="Number of frames to sample per second (default: 1)"
    )
    parser.add_argument(
        "--output-root",
        default="data/output",
        help=(
            "Directory that receives ingestion_manifest.json directly. "
            "Audio and frames are written as siblings of this directory "
            "under data/audio/ and data/frames/. "
            "(default: data/output)"
        )
    )

    args = parser.parse_args()

    ingest_video(
        Path(args.input),
        Path(args.output_root),
        args.sample_fps
    )


if __name__ == "__main__":
    main()
