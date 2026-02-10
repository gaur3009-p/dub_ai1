from services.asr.whisper_asr import WhisperASR
from services.translation.nllb_translator import NLLBTranslator
from services.tts.bark_cloner import BarkVoiceCloner
from config.settings import SUPPORTED_ASR_LANGS, NLLB_MODEL
from services.database.conversation_repo import save_conversation
from services.database.qdrant_client import qdrant_client, COLLECTION_NAME
asr = WhisperASR()
translator = NLLBTranslator(NLLB_MODEL)
tts = BarkVoiceCloner()


def _get_speaker_audio(speaker: str):
    """
    Fetch latest enrolled speaker audio from Qdrant
    """
    results = qdrant_client.scroll(
        collection_name=COLLECTION_NAME,
        limit=1,
        with_payload=True
    )[0]

    if not results:
        raise RuntimeError("Speaker not enrolled")

    return results[0].payload["audio_path"]


def process_audio(audio_path: str, spoken_language: str):
    # ---------- ASR ----------
    asr_lang = SUPPORTED_ASR_LANGS[spoken_language]
    text = asr.transcribe(audio_path, asr_lang)

    if not text or not text.strip():
        raise ValueError("ASR returned empty text")

    # ---------- Language Routing ----------
    if spoken_language == "english":
        src_lang = "eng_Latn"
        tgt_lang = "hin_Deva"
        tts_lang = "hi"
    else:
        src_lang = "hin_Deva"
        tgt_lang = "eng_Latn"
        tts_lang = "en"

    # ---------- Translation ----------
    translated = translator.translate(text, src_lang, tgt_lang)

    # ---------- Fetch Voice ----------
    try:
        speaker_audio = _get_speaker_audio("aditya")
    except Exception as e:
        print("⚠️ Voice not enrolled:", e)
        speaker_audio = None

    # ---------- Bark Voice Cloning ----------
    try:
        audio_out = (
            tts.synthesize(
                text=translated,
                history_prompt=speaker_audio,
                language=tts_lang
            )
            if speaker_audio else None
        )
    except Exception as e:
        print("⚠️ Bark TTS failed:", e)
        audio_out = None

    # ---------- Save Conversation ----------
    try:
        save_conversation(spoken_language, text, translated)
    except Exception as e:
        print("⚠️ Conversation save failed:", e)

    return text, translated, audio_out
