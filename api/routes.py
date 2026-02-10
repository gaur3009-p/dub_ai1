import requests
from services.asr.whisper_asr import WhisperASR
from services.translation.nllb_translator import NLLBTranslator
from services.database.conversation_repo import save_conversation
from config.settings import (
    SUPPORTED_ASR_LANGS,
    NLLB_MODEL,
    VOICE_SERVICE_URL
)

asr = WhisperASR()
translator = NLLBTranslator(NLLB_MODEL)


def enroll_voice_http(audio_path: str, speaker: str):
    with open(audio_path, "rb") as f:
        requests.post(
            f"{VOICE_SERVICE_URL}/enroll",
            params={"speaker": speaker},
            files={"audio": f},
            timeout=60
        )


def synthesize_http(text: str, speaker: str, language: str):
    r = requests.post(
        f"{VOICE_SERVICE_URL}/synthesize",
        params={
            "speaker": speaker,
            "text": text,
            "language": language
        },
        timeout=120
    )
    r.raise_for_status()
    return r.json()["audio_path"]


def process_audio(audio_path: str, spoken_language: str):
    # -------- ASR --------
    text = asr.transcribe(
        audio_path,
        SUPPORTED_ASR_LANGS[spoken_language]
    )

    if not text.strip():
        raise ValueError("Empty ASR output")

    # -------- ROUTING --------
    if spoken_language == "english":
        src_lang, tgt_lang, tts_lang = "eng_Latn", "hin_Deva", "hi"
    else:
        src_lang, tgt_lang, tts_lang = "hin_Deva", "eng_Latn", "en"

    # -------- TRANSLATION --------
    translated = translator.translate(text, src_lang, tgt_lang)

    # -------- VOICE CLONING --------
    audio_out = synthesize_http(
        translated,
        speaker="aditya",
        language=tts_lang
    )

    # -------- SAVE --------
    try:
        save_conversation(spoken_language, text, translated)
    except Exception:
        pass

    return text, translated, audio_out
