
from pathlib import Path
import subprocess
import sys

# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "data" / "input"
OUTPUT_FILE = OUTPUT_DIR / "test_video.mp4"

AUDIO_FILES = [
    Path("/tmp/scene1.wav"),
    Path("/tmp/scene2.wav"),
    Path("/tmp/scene3.wav"),
]

WIDTH = 1280
HEIGHT = 720
FPS = 25
SCENE_DURATION = 10

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

# Explicit hexadecimal colors avoid unsupported color names.
SCENE_BACKGROUNDS = [
    "0x87CEEB",  # Sky blue
    "0xD3D3D3",  # Light gray
    "0xE6E6FA",  # Lavender
]

# --------------------------------------------------
# Scene visual designs
# --------------------------------------------------

SCENE_FILTERS = [
    # Scene 1: Outdoor park
    (
        "drawbox=x=0:y=580:w=1280:h=140:color=0x228B22:t=fill,"
        "drawbox=x=100:y=250:w=35:h=280:color=0x8B4513:t=fill,"
        "drawbox=x=35:y=170:w=165:h=165:color=0x006400:t=fill,"
        "drawbox=x=600:y=330:w=18:h=230:color=black:t=fill,"
        "drawbox=x=560:y=350:w=100:h=18:color=black:t=fill,"
        "drawbox=x=600:y=400:w=18:h=140:color=black:t=fill,"
        "drawbox=x=560:y=540:w=90:h=18:color=black:t=fill,"
        "drawtext=fontfile='{FONT}':text='SCENE 1 - OUTDOOR PARK':"
        "fontsize=42:fontcolor=black:x=40:y=35,"
        "drawtext=fontfile='{FONT}':text='A person walking in a park':"
        "fontsize=30:fontcolor=black:x=40:y=100"
    ),

    # Scene 2: Software engineer
    (
        "drawbox=x=0:y=650:w=1280:h=70:color=0x808080:t=fill,"
        "drawbox=x=300:y=440:w=680:h=30:color=0x8B4513:t=fill,"
        "drawbox=x=580:y=300:w=250:h=140:color=black:t=fill,"
        "drawbox=x=595:y=315:w=220:h=110:color=0xADD8E6:t=fill,"
        "drawbox=x=610:y=445:w=190:h=15:color=0x808080:t=fill,"
        "drawbox=x=850:y=250:w=45:h=190:color=0x000080:t=fill,"
        "drawtext=fontfile='{FONT}':text='SCENE 2 - SOFTWARE ENGINEER':"
        "fontsize=42:fontcolor=black:x=40:y=35,"
        "drawtext=fontfile='{FONT}':text='An engineer working on a laptop':"
        "fontsize=30:fontcolor=black:x=40:y=100"
    ),

    # Scene 3: Presentation
    (
        "drawbox=x=0:y=650:w=1280:h=70:color=0x808080:t=fill,"
        "drawbox=x=700:y=130:w=440:h=300:color=black:t=fill,"
        "drawbox=x=715:y=145:w=410:h=270:color=white:t=fill,"
        "drawbox=x=770:y=210:w=280:h=15:color=blue:t=fill,"
        "drawbox=x=770:y=260:w=220:h=15:color=blue:t=fill,"
        "drawbox=x=400:y=300:w=35:h=270:color=0x000080:t=fill,"
        "drawbox=x=360:y=260:w=115:h=50:color=orange:t=fill,"
        "drawbox=x=170:y=550:w=220:h=60:color=0x808080:t=fill,"
        "drawbox=x=450:y=550:w=220:h=60:color=0x808080:t=fill,"
        "drawtext=fontfile='{FONT}':text='SCENE 3 - PRESENTATION':"
        "fontsize=42:fontcolor=black:x=40:y=35,"
        "drawtext=fontfile='{FONT}':text='A presenter explaining a technical topic':"
        "fontsize=30:fontcolor=black:x=40:y=100"
    ),
]


def check_inputs():
    """Verify required tools and input files exist."""

    if subprocess.run(
        ["which", "ffmpeg"],
        capture_output=True
    ).returncode != 0:
        raise RuntimeError("FFmpeg is not installed or not in PATH.")

    if not Path(FONT).exists():
        raise FileNotFoundError(f"Font not found: {FONT}")

    for audio_file in AUDIO_FILES:
        if not audio_file.exists():
            raise FileNotFoundError(
                f"Audio file missing: {audio_file}\n"
                "Generate the three eSpeak audio files first."
            )


def build_command():
    """Build the FFmpeg command for the three-scene video."""

    command = ["ffmpeg", "-y"]

    # Three 10-second synthetic video sources.
    for background in SCENE_BACKGROUNDS:
        command.extend([
            "-f", "lavfi",
            "-i",
            f"color=c={background}:s={WIDTH}x{HEIGHT}:r={FPS}:d={SCENE_DURATION}"
        ])

    # Three audio inputs.
    for audio_file in AUDIO_FILES:
        command.extend(["-i", str(audio_file)])

    filters = []

    # Process each video scene.
    for index, scene_filter in enumerate(SCENE_FILTERS):
        scene_filter = scene_filter.replace("{FONT}", FONT)

        filters.append(
            f"[{index}:v]{scene_filter}[v{index}]"
        )

    # Normalize and pad each audio clip to 10 seconds.
    for index in range(3):
        audio_input_index = index + 3

        filters.append(
            f"[{audio_input_index}:a]"
            f"aresample=44100,"
            f"apad=pad_dur={SCENE_DURATION},"
            f"atrim=duration={SCENE_DURATION},"
            f"asetpts=PTS-STARTPTS"
            f"[a{index}]"
        )

    # Join the three video scenes.
    filters.append(
        "[v0][v1][v2]concat=n=3:v=1:a=0[vout]"
    )

    # Join the three corresponding audio segments.
    filters.append(
        "[a0][a1][a2]concat=n=3:v=0:a=1[aout]"
    )

    filter_complex = ";".join(filters)

    command.extend([
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", "[aout]",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        "-r", str(FPS),
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        str(OUTPUT_FILE),
    ])

    return command


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    check_inputs()

    command = build_command()

    print("Generating three-scene test video...")
    print("Scene 1: Outdoor park")
    print("Scene 2: Software engineer")
    print("Scene 3: Presentation")

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as error:
        print("\nFFmpeg failed.")
        print(f"Exit code: {error.returncode}")
        sys.exit(error.returncode)

    print("\nVideo generated successfully:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
