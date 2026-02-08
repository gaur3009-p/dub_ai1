import os
from urllib.parse import urlparse

# ========================
# GENERAL
# ========================
APP_NAME = "DubYou"
ENV = os.getenv("ENV", "dev")

# ========================
# POSTGRES (NeonDB)
# ========================
NEON_DATABASE_URL = os.getenv("NEON_DATABASE_URL")

POSTGRES_ENABLED = False
POSTGRES = {}

if NEON_DATABASE_URL:
    parsed = urlparse(NEON_DATABASE_URL)
    POSTGRES = {
        "host": parsed.hostname,
        "port": parsed.port or 5432,
        "db": parsed.path.lstrip("/"),
        "user": parsed.username,
        "password": parsed.password,
        "sslmode": "require",
    }
    POSTGRES_ENABLED = True

# ========================
# REDIS (Redis Cloud)
# ========================
REDIS = {
    "host": os.getenv("REDIS_HOST"),
    "port": int(os.getenv("REDIS_PORT", 6379)),
    "username": os.getenv("REDIS_USERNAME", "default"),
    "password": os.getenv("REDIS_PASSWORD"),
    "db": 0,
    "ssl": True,
}

# ========================
# QDRANT (Cloud)
# ========================
QDRANT = {
    "url": os.getenv("QDRANT_URL"),
    "api_key": os.getenv("QDRANT_API_KEY"),
    "collection": os.getenv("QDRANT_COLLECTION", "voice_embeddings"),
}

# ========================
# ASR (Whisper)
# ========================
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")

SUPPORTED_ASR_LANGS = {
    "english": "en",
    "hindi": "hi",
}

# ========================
# NLLB TRANSLATION
# ========================
NLLB_MODEL = os.getenv(
    "NLLB_MODEL",
    "facebook/nllb-200-distilled-600M"
)

NLLB_LANG_MAP = {
    "english": "eng_Latn",
    "hindi": "hin_Deva",
}

# ========================
# VOICE CLONING (XTTS)
# ========================
XTTS_MODEL_NAME = os.getenv(
    "XTTS_MODEL_NAME",
    "coqui/XTTS-v2"
)

XTTS_LANGUAGES = {
    "english": "en",
    "hindi": "hi",
    "french": "fr",
    "spanish": "es",
}

# ========================
# AUDIO
# ========================
AUDIO_SAMPLE_RATE = 16000
TTS_OUTPUT_SAMPLE_RATE = 24000
