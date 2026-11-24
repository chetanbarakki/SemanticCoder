
import argparse
import json
import re
import time
from pathlib import Path

import ollama


MODEL = "qwen2.5vl:3b"


def load_json(path):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def extract_json(text):
    """Extract structured JSON from the model response."""
    text = text.strip()

    # Handle responses wrapped in Markdown code fences.
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    return json.loads(text)


def analyze_keyframe(keyframe):
    image_path = Path(keyframe["keyframe_path"])

    if not image_path.is_file():
        raise FileNotFoundError(f"Keyframe not found: {image_path}")

    prompt = """
Analyze this image for a semantic video representation.

Return ONLY valid JSON with these fields:
{
    "description": "Concise description of the image",
    "visible_entities": ["visible objects or entities"],
    "activities": ["visible actions or activities"],
    "environment": "Description of the background and setting"
}

Rules:
- Describe only what is actually visible.
- Do not invent people, objects, or activities.
- If an activity cannot be established visually, say so.
- If the image contains text or a presentation slide, mention it.
- Keep the description concise and factual.
"""

    start = time.perf_counter()

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [str(image_path.resolve())]
            }
        ],
        options={
            "temperature": 0,
            "num_predict": 200
        },
        keep_alive="10m"
    )

    elapsed = time.perf_counter() - start
    response_text = response["message"]["content"]

    try:
        semantic_data = extract_json(response_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Model returned invalid JSON for scene "
            f"{keyframe['scene_id']}:\n{response_text}"
        ) from exc

    required_fields = [
        "description",
        "visible_entities",
        "activities",
        "environment"
    ]

    for field in required_fields:
        if field not in semantic_data:
            raise ValueError(
                f"Missing field '{field}' in scene "
                f"{keyframe['scene_id']} response"
            )

    return {
        "scene_id": keyframe["scene_id"],
        "start_seconds": keyframe["scene_start_seconds"],
        "end_seconds": keyframe["scene_end_seconds"],
        "timestamp_seconds": keyframe["timestamp_seconds"],
        "keyframe_path": str(image_path),
        **semantic_data,
        "inference_time_seconds": round(elapsed, 3)
    }


def process_keyframes(input_path, output_path):
    keyframe_data = load_json(input_path)

    results = []
    total_start = time.perf_counter()

    print(f"Processing {keyframe_data['total_keyframes']} keyframes")
    print(f"Model: {MODEL}\n")

    for keyframe in keyframe_data["keyframes"]:
        print(f"Analyzing scene {keyframe['scene_id']}...")

        result = analyze_keyframe(keyframe)
        results.append(result)

        print(
            f"Completed in {result['inference_time_seconds']:.2f}s"
        )
        print(f"Description: {result['description']}\n")

    total_elapsed = time.perf_counter() - total_start

    output = {
        "source_video": keyframe_data["source_video"],
        "model": MODEL,
        "total_scenes": len(results),
        "total_inference_time_seconds": round(total_elapsed, 3),
        "average_inference_time_seconds": round(
            total_elapsed / len(results), 3
        ) if results else 0,
        "scenes": results
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(output, indent=4),
        encoding="utf-8"
    )

    print("Scene understanding completed.")
    print(f"Total time: {total_elapsed:.2f}s")
    print(f"Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate semantic descriptions of video scenes."
    )

    parser.add_argument(
        "--input",
        default="data/output/keyframes.json"
    )

    parser.add_argument(
        "--output",
        default="data/output/scene_understanding.json"
    )

    args = parser.parse_args()

    process_keyframes(args.input, args.output)


if __name__ == "__main__":
    main()
