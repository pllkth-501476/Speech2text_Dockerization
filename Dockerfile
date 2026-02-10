FROM python:3.10-slim

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps:
# - ffmpeg: runtime
# - pkg-config + build-essential + libav*dev: for building PyAV if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    pkg-config \
    libsndfile1 \
    libsndfile1-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavfilter-dev \
    libavformat-dev \
    libavutil-dev \
    libswresample-dev \
    libswscale-dev \
 && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY trascrizione.py confronto.py /app/

RUN mkdir -p /app/input_videos /app/output /app/models /app/logs

VOLUME ["/app/input_videos", "/app/output", "/app/models", "/app/logs"]

CMD ["python", "trascrizione.py", "--no-gui"]
