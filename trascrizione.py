import os
import json
import argparse
import logging
from pathlib import Path

import ffmpeg
from faster_whisper import WhisperModel

# NOTE:
# - Speaker diarization is optional and depends on a gated Hugging Face model.
# - When ENABLE_DIARIZATION=True, the user must provide HF_TOKEN and accept the model terms on Hugging Face.
#   See README.txt for details.

CONFIG_PATH = Path("/app/config.json")

DEFAULT_CONFIG = {
    "MODEL_NAME": "medium",
    "LANGUAGE": "auto",          # 'auto' -> Whisper auto-detection (language=None)
    "TRANSLATE": False,          # if True, uses task='translate'
    "ENABLE_DIARIZATION": False,
    "DIARIZATION_MODEL": "pyannote/speaker-diarization",
    "OUTPUT_FORMAT": "txt",
    "NUM_THREADS": 8,
    "LOG_LEVEL": "INFO",
    "INPUT_FILE": "input.mp4",
    "OUTPUT_FILE": "output.txt",
}

def _str_to_bool(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")

def load_config() -> dict:
    # Start from defaults
    cfg = dict(DEFAULT_CONFIG)

    # Merge config.json if present
    if CONFIG_PATH.exists():
        try:
            cfg.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
        except Exception as e:
            print(f"[WARNING] Failed to parse {CONFIG_PATH}: {e}")

    # Environment overrides (highest priority)
    for key in list(DEFAULT_CONFIG.keys()) + ["HF_TOKEN"]:
        env_val = os.getenv(key)
        if env_val is None:
            continue
        cfg[key] = env_val

    # Type normalization
    cfg["ENABLE_DIARIZATION"] = _str_to_bool(cfg.get("ENABLE_DIARIZATION", False))
    cfg["TRANSLATE"] = _str_to_bool(cfg.get("TRANSLATE", False))
    cfg["NUM_THREADS"] = int(cfg.get("NUM_THREADS", 8))
    cfg["LOG_LEVEL"] = str(cfg.get("LOG_LEVEL", "INFO")).upper()

    return cfg

def setup_logging(level: str) -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    return logging.getLogger("stt")

def extract_audio(input_file: Path, output_wav: Path, logger: logging.Logger) -> Path:
    """Extract mono 16kHz WAV from any audio/video input using ffmpeg."""
    logger.info("Extracting audio -> %s", output_wav)
    (
        ffmpeg
        .input(str(input_file))
        .output(str(output_wav), ac=1, ar="16000", format="wav")
        .overwrite_output()
        .run(quiet=True)
    )
    return output_wav

def seconds_to_timestamp(sec: float) -> str:
    # HH:MM:SS.mmm style
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02}:{m:02}:{s:06.3f}"

def run_transcription(temp_wav: Path, output_file: Path, cfg: dict, logger: logging.Logger,
                      language_override: str | None, task_override: str | None) -> None:
    model_name = cfg["MODEL_NAME"]

    # Decide language
    language_cfg = language_override if language_override is not None else cfg.get("LANGUAGE", "auto")
    language = None if str(language_cfg).lower() == "auto" else str(language_cfg)

    # Decide task (transcribe vs translate)
    if task_override is not None:
        task = task_override
    else:
        task = "translate" if cfg.get("TRANSLATE", False) else "transcribe"

    logger.info("Loading Whisper model: %s (CPU, int8)", model_name)
    model = WhisperModel(model_name, device="cpu", compute_type="int8")

    logger.info("Transcribing (task=%s, language=%s) ...", task, language or "auto-detect")
    segments, info = model.transcribe(
        str(temp_wav),
        beam_size=5,
        language=language,
        task=task
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for seg in segments:
            start = seconds_to_timestamp(seg.start)
            end = seconds_to_timestamp(seg.end)
            f.write(f"[{start} - {end}] {seg.text.strip()}\n")

    logger.info("Transcription complete: %s", output_file)

def run_diarization(temp_wav: Path, output_file: Path, cfg: dict, logger: logging.Logger) -> None:
    """Run speaker diarization (who spoke when). Requires HF_TOKEN + gated model access."""
    hf_token = cfg.get("HF_TOKEN") or os.getenv("HF_TOKEN")
    diar_model = cfg.get("DIARIZATION_MODEL", "pyannote/speaker-diarization")

    if not hf_token:
        logger.warning("HF_TOKEN not provided. Skipping diarization.")
        return

    try:
        from pyannote.audio import Pipeline
    except Exception as e:
        logger.error("pyannote.audio import failed: %s", e)
        return

    logger.info("Loading diarization pipeline: %s", diar_model)
    pipeline = Pipeline.from_pretrained(diar_model, use_auth_token=hf_token)

    logger.info("Running diarization ...")
    diarization = pipeline(str(temp_wav))

    diarization_path = Path(output_file).with_name(Path(output_file).stem + "_diarized.txt")
    with open(diarization_path, "w", encoding="utf-8") as df:
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            df.write(f"[{seconds_to_timestamp(segment.start)} - {seconds_to_timestamp(segment.end)}] {speaker}\n")

    logger.info("Diarization saved to %s", diarization_path)

def resolve_io(cfg: dict, cli_input: str | None, cli_output: str | None) -> tuple[Path, Path]:
    """Resolve input/output without changing code: CLI > env > config.json > defaults."""
    # INPUT_FILE / OUTPUT_FILE allow users to switch files without editing code/container
    input_file = cli_input or os.getenv("INPUT_FILE") or cfg.get("INPUT_FILE") or DEFAULT_CONFIG["INPUT_FILE"]
    output_file = cli_output or os.getenv("OUTPUT_FILE") or cfg.get("OUTPUT_FILE") or DEFAULT_CONFIG["OUTPUT_FILE"]

    # Allow passing absolute paths OR file names relative to the mounted folders
    input_path = Path(input_file)
    if not input_path.is_absolute():
        input_path = Path("/app/input_videos") / input_path

    output_path = Path(output_file)
    if not output_path.is_absolute():
        output_path = Path("/app/output") / output_path

    return input_path, output_path

def main():
    parser = argparse.ArgumentParser(description="Speech-to-Text (Whisper) + optional speaker diarization (pyannote).")
    parser.add_argument("--input", help="Input file path OR filename inside /app/input_videos.")
    parser.add_argument("--output", help="Output file path OR filename inside /app/output.")
    parser.add_argument("--language", default=None, help="Language code (e.g., 'it', 'en') or 'auto'.")
    parser.add_argument("--task", default=None, choices=["transcribe", "translate"],
                        help="Whisper task: transcribe or translate (to English).")
    parser.add_argument("--diarization", action="store_true", help="Force-enable diarization for this run.")
    parser.add_argument("--no-gui", action="store_true", help="CLI mode (no GUI).")
    args = parser.parse_args()

    cfg = load_config()
    logger = setup_logging(cfg.get("LOG_LEVEL", "INFO"))
    logger.info("Loaded configuration from %s (if present)", CONFIG_PATH)
    logger.info("Active configuration: %s", {k: ("***" if k == "HF_TOKEN" and cfg.get(k) else cfg.get(k)) for k in cfg})

    # Respect CLI overrides
    input_path, output_path = resolve_io(cfg, args.input, args.output)

    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        raise SystemExit(1)

    # Extract audio (always to a deterministic location inside mounted output)
    temp_wav = Path("/app/output/temp.wav")
    extract_audio(input_path, temp_wav, logger)

    # Transcription
    run_transcription(
        temp_wav=temp_wav,
        output_file=output_path,
        cfg=cfg,
        logger=logger,
        language_override=args.language,
        task_override=args.task
    )

    # Diarization (config or CLI)
    enable_diar = bool(cfg.get("ENABLE_DIARIZATION", False)) or bool(args.diarization)
    if enable_diar:
        try:
            run_diarization(temp_wav, output_path, cfg, logger)
        except Exception as e:
            logger.warning("Diarization failed or skipped: %s", e)
    else:
        logger.info("Diarization disabled.")

if __name__ == "__main__":
    main()
