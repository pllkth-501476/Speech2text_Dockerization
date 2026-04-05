import os
import json
import argparse
import logging
from pathlib import Path

from faster_whisper import WhisperModel
from tqdm import tqdm
import ffmpeg

# ---------------------- CONFIG & LOGGING ---------------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)

CONFIG_PATH = Path("/app/config.json")

DEFAULT_CONFIG = {
    "MODEL_NAME": "medium",
    "LANGUAGE": "auto",          # 'auto' -> Whisper auto-detection
    "TRANSLATE": False,          # if True, use task='translate'
    "ENABLE_DIARIZATION": False,
    "DIARIZATION_MODEL": "pyannote/speaker-diarization",
    "OUTPUT_FORMAT": "txt",
    "NUM_THREADS": 8,
    "LOG_LEVEL": "INFO",
    "HF_TOKEN": None,
    "INPUT_FILE": "input.mp4",
    "OUTPUT_FILE": "output.txt"
}


def str_to_bool(v):
    return str(v).lower() in ("1", "true", "yes")


def load_config():
    config = DEFAULT_CONFIG.copy()

    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            file_config = json.load(f)
            config.update(file_config)
            logger.info(f"Loaded configuration from {CONFIG_PATH}")

    # Override with env vars if present
    for key in config.keys():
        env_val = os.getenv(key)
        if env_val is not None:
            config[key] = env_val

    # Type normalization
    config["ENABLE_DIARIZATION"] = str_to_bool(config.get("ENABLE_DIARIZATION", False))
    config["TRANSLATE"] = str_to_bool(config.get("TRANSLATE", False))
    config["NUM_THREADS"] = int(config.get("NUM_THREADS", 8))

    logger.info(f"Active configuration: {config}")
    return config


# ---------------------- AUDIO PREPARATION ---------------------- #

def prepare_audio(input_file: Path, wav_output: Path, logger):
    """
    Converts any input media (video or audio) to mono 16kHz WAV.
    Works for mp4, wav, mp3, m4a, etc.
    """
    try:
        logger.info("Preparing audio -> %s", wav_output)
        (
            ffmpeg
            .input(str(input_file))
            .output(str(wav_output), ac=1, ar="16000", format="wav")
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        return wav_output
    except ffmpeg.Error as e:
        logger.error("FFmpeg failed while processing: %s", input_file)
        if e.stderr:
            logger.error(e.stderr.decode())
        raise


# ---------------------- TRANSCRIPTION ---------------------- #

def transcribe_file(input_file: Path, output_file: Path, config, cli_language=None, cli_task=None, cli_diarization=False):
    model_name = config["MODEL_NAME"]

    # Language logic
    cfg_language = config.get("LANGUAGE", "auto")
    language = cli_language if cli_language is not None else cfg_language
    if str(language).lower() == "auto":
        language = None

    # Task logic
    if cli_task is not None:
        task = cli_task
    else:
        task = "translate" if config.get("TRANSLATE", False) else "transcribe"

    enable_diarization = cli_diarization or config["ENABLE_DIARIZATION"]
    hf_token = config.get("HF_TOKEN") or os.getenv("HF_TOKEN")
    diarization_model = config.get("DIARIZATION_MODEL", "pyannote/speaker-diarization")

    temp_wav = Path("/app/output/temp.wav")

    # Convert any media input to wav
    prepare_audio(input_file, temp_wav, logger)

    logger.info("Loading Whisper model: %s", model_name)
    model = WhisperModel(model_name, device="cpu", compute_type="int8")

    logger.info("Starting transcription (task=%s, language=%s)...", task, language or "auto")

    segments, info = model.transcribe(
        str(temp_wav),
        beam_size=5,
        language=language,
        task=task
    )

    duration = getattr(info, "duration", None)

    progress_bar = None
    last_time = 0.0

    if duration:
        progress_bar = tqdm(total=duration, unit="sec", desc="Transcribing")

    with open(output_file, "w", encoding="utf-8") as f:
        for segment in segments:
            start = segment.start
            end = segment.end
            text = segment.text.strip()

            f.write(f"[{start:0>8.2f} - {end:0>8.2f}] {text}\n")

            if progress_bar:
                delta = max(0.0, end - last_time)
                progress_bar.update(delta)
                last_time = end

    if progress_bar:
        progress_bar.close()

    logger.info("Transcription saved to %s", output_file)

    # ---------------------- DIARIZATION ---------------------- #
    if enable_diarization:
        if not hf_token:
            logger.warning("HF_TOKEN missing. Skipping diarization.")
            return

        try:
            from pyannote.audio import Pipeline

            logger.info("Loading pyannote diarization model: %s", diarization_model)

            pipeline = Pipeline.from_pretrained(
                diarization_model,
                use_auth_token=hf_token
            )

            diarization = pipeline(temp_wav)

            diarization_output = Path(output_file).with_name(
                Path(output_file).stem + "_diarized.txt"
            )

            with open(diarization_output, "w", encoding="utf-8") as f:
                for segment, _, speaker in diarization.itertracks(yield_label=True):
                    f.write(
                        f"[{segment.start:0>8.2f} - {segment.end:0>8.2f}] Speaker {speaker}\n"
                    )

            logger.info("Diarization saved to %s", diarization_output)

        except Exception as e:
            logger.error("Diarization failed: %s", e)

    else:
        logger.info("Diarization disabled")


# ---------------------- MAIN ---------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speech-to-Text Pipeline")

    parser.add_argument("--input", help="Input media file (video or audio)")
    parser.add_argument("--output", help="Output transcript file")
    parser.add_argument("--language", help="Language code, e.g. 'en', 'it', or 'auto'")
    parser.add_argument("--task", choices=["transcribe", "translate"], help="Whisper task")
    parser.add_argument("--diarization", action="store_true", help="Enable diarization for this run")
    parser.add_argument("--no-gui", action="store_true", help="Ignored flag kept for compatibility")

    args = parser.parse_args()

    config = load_config()

    input_file = Path(args.input or f"/app/input_videos/{config['INPUT_FILE']}")
    output_file = Path(args.output or f"/app/output/{config['OUTPUT_FILE']}")

    if not input_file.exists():
        logger.error("Input file not found: %s", input_file)
        exit(1)

    transcribe_file(
        input_file,
        output_file,
        config,
        cli_language=args.language,
        cli_task=args.task,
        cli_diarization=args.diarization
    )