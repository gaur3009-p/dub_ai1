FROM python:3.10-slim

# System deps for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    espeak-ng \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY main.py .

# Pre-download YourTTS model at build time (optional – remove to download at runtime)
RUN python -c "from TTS.api import TTS; TTS('tts_models/multilingual/multi-dataset/your_tts')" \
    || echo "Model download will happen at runtime"

ENV USE_GPU=0
EXPOSE 8001

CMD ["python", "main.py"]
