import os
import json
import argparse
import logging
from pathlib import Path

import ffmpeg
from faster_whisper import WhisperModel

# -------------------- logging -------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)

# -------------------- config -------------------- #
CONFIG_PATH = "/app/config.json"

DEFAULT_CONFIG = {
    "MODEL_NAME": "small",
    "LANGUAGE": "auto",
    "TRANSLATE": False,
    "ENABLE_DIARIZATION": False,
    "DIARIZATION_MODEL": "pyannote/speaker-diarization",
    "INPUT_FILE": "input.mp4",
    "OUTPUT_FILE": "output.txt",
    "HF_TOKEN": None
}


def to_bool(value):
    return str(value).lower() in ["true", "1", "yes"]


def load_config():

    config = DEFAULT_CONFIG.copy()

    if os.path.exists(CONFIG_PATH):

        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:

                file_config = json.load(f)

                config.update(file_config)

                logger.info("Loaded configuration from config.json")

        except Exception as e:

            logger.warning(
                f"Could not read config.json, using defaults: {e}"
            )

    # environment variables override config file
    for key in config:

        env_value = os.getenv(key)

        if env_value is not None:
            config[key] = env_value

    config["TRANSLATE"] = to_bool(config["TRANSLATE"])

    config["ENABLE_DIARIZATION"] = to_bool(
        config["ENABLE_DIARIZATION"]
    )

    return config


# -------------------- audio preparation -------------------- #
def prepare_audio(input_file, output_wav):

    """
    Convert any input (video or audio)
    into mono 16kHz wav
    """

    try:

        logger.info(f"Preparing audio from {input_file}")

        (
            ffmpeg
            .input(str(input_file))
            .output(
                str(output_wav),
                ac=1,
                ar="16000",
                format="wav"
            )
            .overwrite_output()
            .run(
                capture_stdout=True,
                capture_stderr=True
            )
        )

    except ffmpeg.Error as e:

        logger.error("FFmpeg failed.")

        if e.stderr:
            logger.error(
                e.stderr.decode(errors="ignore")
            )

        raise


# -------------------- transcription -------------------- #
def transcribe_file(
    input_file,
    output_file,
    config,
    cli_language=None,
    cli_task=None,
    force_diarization=False
):

    model_name = config["MODEL_NAME"]

    # language
    language = (
        cli_language
        if cli_language
        else config["LANGUAGE"]
    )

    if str(language).lower() == "auto":
        language = None

    # task
    if cli_task:
        task = cli_task
    else:
        task = (
            "translate"
            if config["TRANSLATE"]
            else "transcribe"
        )

    enable_diarization = (
        force_diarization
        or config["ENABLE_DIARIZATION"]
    )

    hf_token = (
        config.get("HF_TOKEN")
        or os.getenv("HF_TOKEN")
    )

    diarization_model = config.get(
        "DIARIZATION_MODEL",
        "pyannote/speaker-diarization"
    )

    temp_wav = Path("/app/output/temp.wav")

    # convert media to wav
    prepare_audio(input_file, temp_wav)

    logger.info(
        f"Loading Whisper model: {model_name}"
    )

    model = WhisperModel(
        model_name,
        device="cpu",
        compute_type="int8"
    )

    logger.info("Starting transcription...")
    logger.info(
        "Transcription started: 0%% completed"
    )

    segments, info = model.transcribe(
        str(temp_wav),
        beam_size=1,
        language=language,
        task=task
    )

    duration = getattr(info, "duration", None)

    with open(output_file, "w", encoding="utf-8") as f:

        for segment in segments:

            start = segment.start
            end = segment.end
            text = segment.text.strip()

            f.write(
                f"[{start:0>8.2f} - "
                f"{end:0>8.2f}] "
                f"{text}\n"
            )

            # progress percentage
            if duration:

                progress = int(
                    (end / duration) * 100
                )

                # avoid values above 100
                if progress > 100:
                    progress = 100

                logger.info(
                    "Transcription progress: %d%% completed",
                    progress
                )

    logger.info(
        f"Transcription saved to {output_file}"
    )

    # -------------------- diarization -------------------- #
    if enable_diarization:

        if not hf_token:

            logger.warning(
                "HF_TOKEN missing. "
                "Skipping diarization."
            )

            return

        try:

            from pyannote.audio import Pipeline

            logger.info(
                "Loading pyannote diarization model: "
                f"{diarization_model}"
            )

            # compatibility for different versions
            try:

                pipeline = Pipeline.from_pretrained(
                    diarization_model,
                    token=hf_token
                )

            except TypeError:

                pipeline = Pipeline.from_pretrained(
                    diarization_model,
                    use_auth_token=hf_token
                )

            diarization = pipeline(str(temp_wav))

            diarization_output = (
                output_file.parent /
                f"{output_file.stem}_diarized.txt"
            )

            with open(
                diarization_output,
                "w",
                encoding="utf-8"
            ) as f:

                for (
                    segment,
                    _,
                    speaker
                ) in diarization.itertracks(
                    yield_label=True
                ):

                    f.write(
                        f"[{segment.start:0>8.2f} - "
                        f"{segment.end:0>8.2f}] "
                        f"Speaker {speaker}\n"
                    )

            logger.info(
                f"Diarization saved to "
                f"{diarization_output}"
            )

        except Exception as e:

            logger.error(
                f"Diarization failed: {e}"
            )

    else:

        logger.info("Diarization disabled.")


# -------------------- main -------------------- #
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Speech-to-Text Pipeline"
    )

    parser.add_argument(
        "--input",
        help="Input media file"
    )

    parser.add_argument(
        "--output",
        help="Output transcript file"
    )

    parser.add_argument(
        "--language",
        help="Language code "
             "(en, it, hi, auto, etc.)"
    )

    parser.add_argument(
        "--task",
        choices=["transcribe", "translate"],
        help="Whisper task"
    )

    parser.add_argument(
        "--diarization",
        action="store_true",
        help="Enable diarization for this run"
    )

    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Compatibility flag"
    )

    args = parser.parse_args()

    config = load_config()

    input_name = (
        args.input
        if args.input
        else f"/app/input_videos/{config['INPUT_FILE']}"
    )

    output_name = (
        args.output
        if args.output
        else f"/app/output/{config['OUTPUT_FILE']}"
    )

    input_file = Path(input_name)
    output_file = Path(output_name)

    if not input_file.exists():

        logger.error(
            f"Input file not found: {input_file}"
        )

        exit(1)

    transcribe_file(
        input_file=input_file,
        output_file=output_file,
        config=config,
        cli_language=args.language,
        cli_task=args.task,
        force_diarization=args.diarization
    )