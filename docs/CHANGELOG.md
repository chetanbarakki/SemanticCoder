# Changelog

All notable changes to semantic-codec are recorded here in plain English.
Most recent entry first.

---

## 2026-11-24

### src/ingestion/semantic_encoder.py — byte-accounting fix + docstring

**What changed.**
`package_manifest.json` (written inside the ZIP) previously recorded
`keyframe_bytes` as the running sum of raw keyframe file sizes on disk
*before* ZIP compression (138 kB on the test clip vs. the 73 kB ZIP on
disk). A consumer parsing that field to learn the package size would see
the wrong number.

The fix uses a two-pass strategy:

1. **Pass 1** — zip all content files (JSON + keyframes), seal the archive,
   measure its size with `os.path.getsize`.  This is the content-only size.
2. **Pass 2** — append `package_manifest.json` with `zip_size_bytes` set to
   the pass-1 measurement, then call `os.path.getsize` again for the final
   on-disk size that goes into `encoding_report.json`.

The old `keyframe_bytes` field was renamed to `keyframe_bytes_uncompressed`
(raw JPEG sum, still useful for data-volume accounting) and kept alongside
the new `zip_size_bytes`.  `encoding_report.json` field `keyframe_data_bytes`
was similarly renamed to `keyframe_data_bytes_uncompressed` — additive change,
no old fields removed.  `package_version` bumped from `"0.1"` to `"0.2"`.

A module docstring (inputs, outputs, failure mode) was added per project
rules.

---

### config.yaml — new file at project root

**What changed.**
A single `config.yaml` now holds all tuneable knobs: data paths, `sample_fps`,
scene-detection threshold, YOLO model / confidence / device, Whisper model /
device / compute type, VLM model / keep-alive, and decode FPS.  All values
match the existing `argparse` defaults so no existing behaviour changes.
`scripts/run_pipeline.py` reads this file; individual modules still accept CLI
flags that override it.

---

### scripts/run_pipeline.py — new file

**What changed.**
End-to-end pipeline runner for a single input video.  Calls each stage's
Python function directly (no subprocess), so import overhead is paid once and
GPU memory stays warm between stages.  Every stage is wrapped in `StageTimer`
which records wall-clock seconds, peak VRAM in MB via `pynvml` (gracefully
skipped if no NVIDIA driver is present, recorded as `null`), and peak process
RSS in MB via `psutil`.  Stages timed: ingestion, scene\_detection, whisper,
yolo, keyframe\_extraction, vlm, semantic\_representation, packaging, decode.
Results are written to `results/<clip_name>/metrics.json` alongside the byte
totals copied from `encoding_report.json`.

---

### tests/test_semantic_encoder.py — new file

**What changed.**
Five unit tests covering the byte-accounting fix.  Uses a synthetic 1×1 JPEG
and a minimal in-memory `semantic_representation.json`; no GPU or real video
required.  Tests verify: `zip_size_bytes` is positive and less than the final
on-disk size; `encoding_report.encoded_size_bytes` matches `os.path.getsize`
on the sealed archive; `package_version` is `"0.2"`; `keyframe_bytes_uncompressed`
is a positive integer; ZIP contains the expected member names.  All 5 pass.
