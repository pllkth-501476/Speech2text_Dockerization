# 🎙️ Speech-to-Text Pipeline (Dockerized)

This project implements a **containerized speech-to-text pipeline** using the Whisper model.
It supports both **audio and video inputs**, performs automatic transcription with timestamps, and optionally enables **speaker diarization**.

The entire system is packaged using Docker to ensure **reproducibility, portability, and ease of use**.

---

## 🚀 Features

* Supports **audio and video inputs**
* Works with common formats:

  * Audio: `.wav`, `.mp3`, `.m4a`
  * Video: `.mp4`, `.mkv`, `.avi`
* Automatic **audio extraction and normalization** using FFmpeg
* High-quality transcription using **faster-whisper**
* **Timestamped output**
* Optional **speaker diarization** using pyannote.audio
* Runtime configuration via:

  * `config.json`
  * environment variables
  * command-line arguments
* Progress display during transcription
* Fully reproducible via Docker

---

## 🧠 Pipeline Overview

```
Input File (Audio/Video)
        │
        ▼
FFmpeg (Audio Extraction & Conversion)
        │
        ▼
Standard WAV (mono, 16kHz)
        │
        ▼
Whisper Model (Transcription)
        │
        ▼
Timestamped Text Output
        │
        ├──────────────► output.txt
        │
        ▼
(Optional) Speaker Diarization
        │
        ▼
Speaker Segments Output
        └──────────────► output_diarized.txt
```

---

## 📋 Prerequisites

* Docker (Docker Desktop or Docker Engine)
* Internet connection (required for first model download)
* Optional:

  * Hugging Face account + token (for diarization)

---

## 📁 Project Structure

```
Speech_to_Text_Containerization_CPU/
│
├── trascrizione.py        # main transcription pipeline
├── confronto.py           # evaluation script
├── Dockerfile             # container setup
├── docker-compose.yml     # container execution
├── config.json            # configuration file
├── requirements.txt       # dependencies
│
├── input_videos/          # input files
├── output/                # generated outputs
```

---

## ▶️ How to Run

### 1. Create required folders

```bash
mkdir input_videos output
```

---

### 2. Add input file

Place any file inside:

```
input_videos/
```

Examples:

```
input_videos/input.mp4
input_videos/lecture.wav
input_videos/speech.mp3
```

---

### 3. Run the pipeline

```bash
docker compose up --build
```

---

### 4. Check output

```
output/output.txt
```

---

## ⚙️ Runtime Configuration

### 🔹 Using environment variables

```bash
INPUT_FILE=lecture.wav OUTPUT_FILE=lecture.txt docker compose up
```

---

### 🔹 Using command-line arguments

```bash
docker compose run --rm speech2text \
python trascrizione.py \
--input /app/input_videos/lecture.wav \
--output /app/output/lecture.txt
```

---

### 🔹 Using config.json

```json
{
  "INPUT_FILE": "input.mp4",
  "OUTPUT_FILE": "output.txt",
  "ENABLE_DIARIZATION": false
}
```

---

## 🎧 Example: Audio Input (Professor Requirement)

```bash
INPUT_FILE=lecture.wav OUTPUT_FILE=lecture.txt docker compose up
```

✔ No video required
✔ Direct audio transcription supported

---

## 👥 Speaker Diarization (Optional)

To enable diarization:

1. Create a Hugging Face account
2. Accept model access:
   https://huggingface.co/pyannote/speaker-diarization
3. Generate a token

Run:

```bash
export HF_TOKEN=your_token_here
ENABLE_DIARIZATION=true docker compose up
```

---

## 🐳 Run via Docker Hub

Instead of cloning the repo:

```bash
docker run --rm \
  -v $(pwd)/input_videos:/app/input_videos \
  -v $(pwd)/output:/app/output \
  kavitha245p/speech2text:cpu
```

---

## 📊 Output

* `output.txt` → transcription with timestamps
* `*_diarized.txt` → speaker-separated output (if enabled)

---

## ⚠️ Known Limitations

* Speaker diarization requires a valid Hugging Face token
* First run may be slow due to model download
* Long audio files require more CPU and memory
* Progress bar is based on processed segments (not continuous from 0%)

---

## 🎯 Use Case

This project was developed as part of a Master's thesis to convert a research prototype into a **reproducible and portable speech-to-text system using Docker**.

---

## 🔗 Links

* GitHub Repository:
  https://github.com/pllkth-501476/Speech2text_Dockerization

* Docker Hub Image:
  https://hub.docker.com/r/kavitha245p/speech2text

---

## 👩‍💻 Author

Kavitha
