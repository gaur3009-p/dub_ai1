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
WHISPER_MODEL = "base"

SUPPORTED_ASR_LANGS = {
    "english": "en",
    "hindi": "hi",
}

# ========================
# NLLB TRANSLATION
# ========================
NLLB_MODEL = "facebook/nllb-200-distilled-600M"

NLLB_LANG_MAP = {
    "english": "eng_Latn",
    "hindi": "hin_Deva",
}

# ========================
# MMS – MULTILINGUAL TTS
# ========================
MMS_TTS_BASE = "facebook/mms-tts"

MMS_TTS_LANGUAGES = {
    "english": "eng",
    "hindi": "hin",
}

DEFAULT_TTS_LANG = "eng"
