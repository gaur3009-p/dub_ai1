import os
from urllib.parse import urlparse

APP_NAME = "DubYou"
ENV = os.getenv("ENV", "dev")

# ------------------------
# DATABASES
# ------------------------
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

REDIS = {
    "host": os.getenv("REDIS_HOST"),
    "port": int(os.getenv("REDIS_PORT", 6379)),
    "username": os.getenv("REDIS_USERNAME", "default"),
    "password": os.getenv("REDIS_PASSWORD"),
    "db": 0,
    "ssl": True,
}

# ------------------------
# ASR / TRANSLATION
# ------------------------
WHISPER_MODEL = "base"

SUPPORTED_ASR_LANGS = {
    "english": "en",
    "hindi": "hi",
}

NLLB_MODEL = "facebook/nllb-200-distilled-600M"

NLLB_LANG_MAP = {
    "english": "eng_Latn",
    "hindi": "hin_Deva",
}

# ------------------------
# VOICE SERVICE
# ------------------------
VOICE_SERVICE_URL = os.getenv(
    "VOICE_SERVICE_URL",
    "http://localhost:8001"
)
