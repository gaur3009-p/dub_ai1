from services.asr.whisper_asr import WhisperASR
from services.translation.nllb_translator import NLLBTranslator
from services.tts.voice_synthesizer import VoiceSynthesizer
from config.settings import (
    SUPPORTED_ASR_LANGS,
    NLLB_LANG_MAP,
    NLLB_MODEL
)
from services.database.conversation_repo import save_conversation
from services.database.qdrant_client import qdrant_client, COLLECTION_NAME

asr = WhisperASR()
translator = NLLBTranslator(NLLB_MODEL)
tts = VoiceSynthesizer()


def _get_speaker_embedding(speaker: str):
    """
    Fetch latest enrolled speaker embedding from Qdrant
    """
    search_result = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=[0.0] * 40,  # dummy vector for payload-only search
        limit=1,
        with_payload=True
    )

    if not search_result:
        raise RuntimeError("Speaker not enrolled. Please enroll voice first.")

    return search_result[0].vector


def process_audio(audio_path: str, spoken_language: str):
    # ---------------- ASR ----------------
    asr_lang = SUPPORTED_ASR_LANGS[spoken_language]
    text = asr.transcribe(audio_path, asr_lang)

    if not text or not text.strip():
        raise ValueError("ASR returned empty text")

    # ---------------- Language Routing ----------------
    if spoken_language == "english":
        src_lang = "eng_Latn"
        tgt_lang = "hin_Deva"
        tts_lang = "hi"
    else:
        src_lang = "hin_Deva"
        tgt_lang = "eng_Latn"
        tts_lang = "en"

    # ---------------- Translation ----------------
    translated = translator.translate(text, src_lang, tgt_lang)

    # ---------------- Speaker Embedding ----------------
    try:
        speaker_embedding = _get_speaker_embedding("aditya")
    except Exception as e:
        print("⚠️ Speaker fetch failed:", e)
        speaker_embedding = None

    # ---------------- Voice Cloning TTS ----------------
    try:
        audio_out = (
            tts.synthesize(translated, speaker_embedding, tts_lang)
            if speaker_embedding is not None
            else None
        )
    except Exception as e:
        print("⚠️ TTS failed:", e)
        audio_out = None

    # ---------------- Save Conversation ----------------
    try:
        save_conversation(spoken_language, text, translated)
    except Exception as e:
        print("⚠️ Conversation save failed:", e)

    return text, translated, audio_out
