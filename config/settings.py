import os

# ========================
# GENERAL
# ========================
APP_NAME = "DubYou"
ENV = os.getenv("ENV", "dev")

# ========================
# POSTGRES
# ========================
POSTGRES = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", 5432)),
    "db": os.getenv("POSTGRES_DB", "dubyou"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
}

# ========================
# REDIS
# ========================
REDIS = {
    "host": os.getenv("REDIS_HOST", "localhost"),
    "port": int(os.getenv("REDIS_PORT", 6379)),
    "db": 0,
}

# ========================
# QDRANT
# ========================
QDRANT = {
    "host": os.getenv("QDRANT_HOST", "localhost"),
    "port": int(os.getenv("QDRANT_PORT", 6333)),
    "collection": "voice_embeddings",
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
