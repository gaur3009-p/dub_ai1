# ========================
# VOICE CLONING CONFIG
# ========================

# Supported languages for voice cloning + TTS output
CLONE_SUPPORTED_LANGUAGES = ["english", "hindi", "french", "spanish"]

CLONE_LANG_MAP = {
    "english": {
        "whisper": "en",
        "nllb_src": "eng_Latn",
        "nllb_tgt": "eng_Latn",
        "mms_tts": "eng",
        "yourtts": "en",
    },
    "hindi": {
        "whisper": "hi",
        "nllb_src": "hin_Deva",
        "nllb_tgt": "hin_Deva",
        "mms_tts": "hin",
        "yourtts": "hi",
    },
    "french": {
        "whisper": "fr",
        "nllb_src": "fra_Latn",
        "nllb_tgt": "fra_Latn",
        "mms_tts": "fra",
        "yourtts": "fr-fr",
    },
    "spanish": {
        "whisper": "es",
        "nllb_src": "spa_Latn",
        "nllb_tgt": "spa_Latn",
        "mms_tts": "spa",
        "yourtts": "es",
    },
}

# Speaker Encoder (for voice embedding / enrollment)
SPEAKER_ENCODER_MODEL = "speechbrain/spkrec-ecapa-voxceleb"

# YourTTS multilingual voice clone model
YOURTTS_MODEL = "tts_models/multilingual/multi-dataset/your_tts"

# VITS fine-tune base (per-language fine-tuning)
VITS_BASE_MODELS = {
    "english": "facebook/mms-tts-eng",
    "hindi":   "facebook/mms-tts-hin",
    "french":  "facebook/mms-tts-fra",
    "spanish": "facebook/mms-tts-spa",
}

# Training settings
CLONE_TRAIN_EPOCHS       = 200
CLONE_TRAIN_BATCH_SIZE   = 8
CLONE_TRAIN_LR           = 1e-4
CLONE_MIN_ENROLL_SECONDS = 10   # minimum audio to enroll a voice
CLONE_EMBEDDING_DIM      = 192  # ECAPA-TDNN output dim

# Storage paths
VOICE_PROFILES_DIR = "voice_profiles"    # local fallback
CLONE_CHECKPOINTS_DIR = "checkpoints"
