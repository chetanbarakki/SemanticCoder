
import argparse
import json
from pathlib import Path

from faster_whisper import WhisperModel


def transcribe_audio(
    audio_path,
    output_path,
    model_size="small.en",
    device="cuda",
    compute_type="float16"
):
    audio_path = Path(audio_path)
    output_path = Path(output_path)

    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    print(f"Loading Faster-Whisper model: {model_size}")
    print(f"Device: {device}, compute type: {compute_type}")

    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type
    )

    print("Starting transcription...")

    segments, info = model.transcribe(
        str(audio_path),
        language="en",
        beam_size=5,
        vad_filter=True,
        word_timestamps=True
    )

    result_segments = []

    for segment in segments:
        words = []

        if segment.words:
            for word in segment.words:
                words.append({
                    "word": word.word,
                    "start": round(word.start, 3),
                    "end": round(word.end, 3),
                    "probability": round(word.probability, 4)
                })

        result_segments.append({
            "start": round(segment.start, 3),
            "end": round(segment.end, 3),
            "text": segment.text.strip(),
            "words": words
        })

        print(
            f"[{segment.start:.2f}s - {segment.end:.2f}s] "
            f"{segment.text.strip()}"
        )

    output = {
        "source_audio": str(audio_path),
        "model": model_size,
        "language": info.language,
        "language_probability": round(info.language_probability, 4),
        "duration_seconds": round(info.duration, 3),
        "segments": result_segments
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=4, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\nTranscription completed.")
    print(f"Segments: {len(result_segments)}")
    print(f"Output: {output_path}")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio using Faster-Whisper."
    )

    parser.add_argument(
        "--input",
        default="data/audio/test_video.wav"
    )

    parser.add_argument(
        "--output",
        default="data/output/transcription.json"
    )

    parser.add_argument(
        "--model",
        default="small.en"
    )

    parser.add_argument(
        "--device",
        default="cuda"
    )

    parser.add_argument(
        "--compute-type",
        default="float16"
    )

    args = parser.parse_args()

    transcribe_audio(
        args.input,
        args.output,
        args.model,
        args.device,
        args.compute_type
    )


if __name__ == "__main__":
    main()
