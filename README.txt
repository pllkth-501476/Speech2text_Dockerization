README — Speech-to-Text Containerization (CPU) with Optional Diarization
=====================================================================

Project context
---------------
This project containerizes a Speech-to-Text (STT) pipeline based on Whisper (via faster-whisper),
and optionally performs speaker diarization using pyannote.audio.

The container is CLI-only (no GUI) and is designed to be configurable WITHOUT rebuilding the image.
Configuration is supported via:
- config.json (mounted into the container)
- environment variables in docker-compose.yml (or shell overrides)

Folder structure
----------------
./input_videos/   -> place input MP4 (or audio) files here
./output/         -> generated transcripts and diarization files appear here
./config.json     -> runtime configuration (mounted into /app/config.json)

Quick start (transcription only)
--------------------------------
1) Put an input file into input_videos/
   Example: input_videos/input.mp4

2) Run:
   docker compose up --build

3) Output will be created in:
   output/output.txt   (default)
   output/temp.wav     (intermediate audio extraction)

Changing input/output WITHOUT editing code or rebuilding
--------------------------------------------------------
Option A: Use environment variables (recommended)
  INPUT_FILE=meeting.mp4 OUTPUT_FILE=meeting.txt docker compose up --build

Option B: Edit config.json (mounted volume)
  Set:
    "INPUT_FILE": "meeting.mp4",
    "OUTPUT_FILE": "meeting.txt"

Translation to English
----------------------
You can translate the audio to English using Whisper task="translate".

Option A: Set in config.json:
  "TRANSLATE": true

Option B: Run once via CLI:
  docker compose run --rm speech2text python trascrizione.py --task translate

Language selection
------------------
- Use "LANGUAGE": "auto" for automatic language detection.
- Or set a specific language code, e.g. "it", "en", "fr".

Speaker diarization (optional)
------------------------------
Diarization is enabled only when:
- ENABLE_DIARIZATION=true AND
- a valid Hugging Face token is provided via HF_TOKEN AND
- you have accepted the model terms on Hugging Face.

1) Accept model terms:
   https://huggingface.co/pyannote/speaker-diarization

2) Create a Hugging Face access token (Read):
   https://huggingface.co/settings/tokens

3) Provide the token at runtime (recommended via shell env var):
   export HF_TOKEN=hf_xxx
   ENABLE_DIARIZATION=true docker compose up --build

When diarization runs, you will get:
  output/<output_name>_diarized.txt

Evaluation script (confronto.py)
--------------------------------
This script compares a reference transcript (REF) with a hypothesis transcript (HYP) and writes a CSV report.

Example:
  docker compose run --rm speech2text python confronto.py --ref /app/output/ref.txt --hyp /app/output/hyp.txt --out /app/output/report.csv

Notes
-----
- This container is CPU-only. The Whisper model is loaded with compute_type=int8 to reduce memory usage.
- For long files, consider using a smaller Whisper model (e.g., small) or increasing available RAM/CPU.
