"""
Pipeline Runner — encodes and decodes one video end-to-end, timing every stage
and recording peak hardware utilisation.

Inputs:
    --input   path to a video file  (default: value from config.yaml)
    --config  path to config.yaml   (default: config.yaml at project root)
    --output  override for the results directory (default: results/<clip_name>/)

Outputs:
    results/<clip_name>/metrics.json  – wall-clock seconds per stage, peak VRAM
                                        (MB), peak RSS (MB), and encoding numbers

Failure mode:
    If pynvml cannot initialise (no NVIDIA driver) VRAM sampling is skipped and
    vram_peak_mb is recorded as null rather than raising an exception.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import psutil
import yaml

# ---------------------------------------------------------------------------
# Allow running from any working directory by adding the project root to the
# import path so the src/ modules are reachable without installation.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.video_ingestion import ingest_video
from src.visual.scene_detection import detect_scenes
from src.audio.transcription import transcribe_audio
from src.visual.object_detection import detect_objects
from src.visual.keyframe_extraction import extract_keyframes
from src.visual.scene_understanding import process_keyframes
from src.fusion.semantic_representation import build_semantic_representation
from src.ingestion.semantic_encoder import encode_package
from src.fusion.semantic_decoder import decode_package


# ---------------------------------------------------------------------------
# Hardware sampling helpers
# ---------------------------------------------------------------------------

def _init_nvml():
    """Return an nvml handle or None if NVML is unavailable."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml, handle
    except Exception:
        return None, None


def _vram_used_mb(pynvml, handle):
    """Return current VRAM used in MB, or None."""
    if pynvml is None:
        return None
    try:
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / (1024 ** 2)
    except Exception:
        return None


def _rss_mb():
    """Return the current process RSS in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


# ---------------------------------------------------------------------------
# Timed stage wrapper
# ---------------------------------------------------------------------------

class StageTimer:
    """Runs a callable, records wall time, and tracks peak VRAM / RAM."""

    def __init__(self, pynvml, nvml_handle):
        self._pynvml = pynvml
        self._nvml_handle = nvml_handle
        self.results = {}

    def run(self, name, fn, *args, **kwargs):
        print(f"\n{'='*60}")
        print(f"  STAGE: {name}")
        print(f"{'='*60}")

        vram_before = _vram_used_mb(self._pynvml, self._nvml_handle) or 0.0
        rss_before  = _rss_mb()

        t0 = time.perf_counter()
        fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0

        vram_after = _vram_used_mb(self._pynvml, self._nvml_handle)
        rss_after  = _rss_mb()

        vram_peak = (
            round(max(vram_before, vram_after), 1)
            if vram_after is not None else None
        )

        self.results[name] = {
            "wall_seconds": round(elapsed, 3),
            "vram_peak_mb": vram_peak,
            "rss_peak_mb": round(max(rss_before, rss_after), 1),
        }

        print(
            f"\n  ✓ {name} finished in {elapsed:.2f}s  "
            f"| VRAM peak: {vram_peak} MB  "
            f"| RSS peak: {self.results[name]['rss_peak_mb']} MB"
        )

        return self.results[name]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(video_path: Path, cfg: dict, results_dir: Path):
    clip_name = video_path.stem
    out = results_dir / clip_name
    out.mkdir(parents=True, exist_ok=True)

    # data_out is the single directory all stage outputs land in.
    # It is derived from config.yaml paths.output_dir so changing that
    # key is enough to redirect a whole run (e.g. data/output_real).
    data_out = PROJECT_ROOT / cfg["paths"]["output_dir"]

    # Audio and frames are siblings of data_out (they live under data/).
    # ingest_video derives these same paths as data_out.parent / "audio"
    # and data_out.parent / "frames", so the paths always agree.
    audio_root  = data_out.parent / "audio"
    kf_dir      = PROJECT_ROOT / cfg["paths"]["keyframes_dir"]

    manifest_path      = data_out / "ingestion_manifest.json"
    scenes_path        = data_out / "scenes.json"
    transcription_path = data_out / "transcription.json"
    detections_path    = data_out / "detections.json"
    keyframes_path     = data_out / "keyframes.json"
    understanding_path = data_out / "scene_understanding.json"
    semantic_path      = data_out / "semantic_representation.json"
    package_path       = data_out / "semantic_package.zip"
    reconstructed_path = data_out / "reconstructed_video.mp4"

    pynvml, nvml_handle = _init_nvml()
    timer = StageTimer(pynvml, nvml_handle)

    # 1. Ingestion --------------------------------------------------------
    # Pass data_out as output_root so ingest_video writes the manifest
    # directly into data_out (not into data_out.parent / "output").
    timer.run(
        "ingestion",
        ingest_video,
        video_path,
        data_out,
        cfg["ingestion"]["sample_fps"],
    )

    # 2. Scene detection --------------------------------------------------
    timer.run(
        "scene_detection",
        detect_scenes,
        str(video_path),
        str(scenes_path),
        cfg["scene_detection"]["threshold"],
    )

    # 3. Whisper transcription --------------------------------------------
    audio_path = audio_root / f"{clip_name}.wav"
    timer.run(
        "whisper",
        transcribe_audio,
        str(audio_path),
        str(transcription_path),
        cfg["transcription"]["model"],
        cfg["transcription"]["device"],
        cfg["transcription"]["compute_type"],
    )

    # 4. YOLO object detection -------------------------------------------
    timer.run(
        "yolo",
        detect_objects,
        str(manifest_path),
        str(detections_path),
        cfg["object_detection"]["model"],
        cfg["object_detection"]["device"],
    )

    # 5. Keyframe extraction ---------------------------------------------
    timer.run(
        "keyframe_extraction",
        extract_keyframes,
        str(manifest_path),
        str(scenes_path),
        str(kf_dir),
    )

    # 6. VLM scene understanding -----------------------------------------
    timer.run(
        "vlm",
        process_keyframes,
        str(keyframes_path),
        str(understanding_path),
    )

    # 7. Semantic representation assembly --------------------------------
    timer.run(
        "semantic_representation",
        build_semantic_representation,
        str(scenes_path),
        str(transcription_path),
        str(detections_path),
        str(understanding_path),
        str(keyframes_path),
        str(semantic_path),
    )

    # 8. Packaging (encode) ----------------------------------------------
    timer.run(
        "packaging",
        encode_package,
        str(semantic_path),
        str(package_path),
    )

    # 9. Decode ----------------------------------------------------------
    timer.run(
        "decode",
        decode_package,
        str(package_path),
        str(reconstructed_path),
        cfg["decode"]["fps"],
    )

    # Shut down NVML cleanly.
    if pynvml is not None:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    # -------------------------------------------------------------------
    # Read the encoding report written by encode_package for byte totals.
    # -------------------------------------------------------------------
    encoding_report = {}
    report_path = data_out / "encoding_report.json"
    if report_path.is_file():
        encoding_report = json.loads(
            report_path.read_text(encoding="utf-8")
        )

    # -------------------------------------------------------------------
    # Assemble and write metrics.json
    # -------------------------------------------------------------------
    total_wall = sum(
        s["wall_seconds"] for s in timer.results.values()
    )

    metrics = {
        "clip_name": clip_name,
        "input_video": str(video_path),
        "total_wall_seconds": round(total_wall, 3),
        "stages": timer.results,
        "encoding": {
            "original_size_bytes": encoding_report.get("original_size_bytes"),
            "encoded_size_bytes":  encoding_report.get("encoded_size_bytes"),
            "size_reduction_percent": encoding_report.get("size_reduction_percent"),
            "size_ratio_original_to_package": encoding_report.get(
                "size_ratio_original_to_package"
            ),
        },
    }

    metrics_path = out / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=4),
        encoding="utf-8"
    )

    print(f"\n{'='*60}")
    print(f"  Pipeline complete — total {total_wall:.1f}s")
    print(f"  Metrics: {metrics_path}")
    print(f"{'='*60}\n")

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run the full semantic-codec encode+decode pipeline on one video "
            "and record per-stage timing, VRAM, and RAM into metrics.json."
        )
    )
    parser.add_argument(
        "--input",
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config.yaml"),
        help="Path to config.yaml (default: <project_root>/config.yaml).",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help=(
            "Directory that receives results/<clip_name>/metrics.json. "
            "Defaults to the value of paths.results_dir in config.yaml."
        ),
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_file():
        parser.error(f"Config file not found: {config_path}")

    with open(config_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    # Resolve the results directory: CLI flag > config.yaml > hardcoded default.
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        results_dir = PROJECT_ROOT / cfg["paths"].get("results_dir", "results")

    # Resolve the input video: CLI flag > config.yaml default input.
    if args.input:
        video_path = Path(args.input)
    else:
        default_input = (
            PROJECT_ROOT
            / cfg["paths"]["input_dir"]
            / "test_video.mp4"
        )
        video_path = default_input

    if not video_path.is_file():
        parser.error(f"Input video not found: {video_path}")

    run_pipeline(video_path.resolve(), cfg, results_dir.resolve())


if __name__ == "__main__":
    main()
