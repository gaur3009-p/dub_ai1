from services.asr.whisper_asr import WhisperASR
from services.translation.nllb_translator import NLLBTranslator
from services.tts.voice_synthesizer import VoiceSynthesizer
from config.settings import (
    SUPPORTED_ASR_LANGS,
    NLLB_LANG_MAP,
    MMS_TTS_LANGUAGES,
    NLLB_MODEL
)
from services.voice_identity.voice_storage import store_voice
from services.database.conversation_repo import save_conversation

asr = WhisperASR()
translator = NLLBTranslator(NLLB_MODEL)
tts = VoiceSynthesizer()

def process_audio(audio_path: str, spoken_language: str):
    asr_lang = SUPPORTED_ASR_LANGS[spoken_language]
    text = asr.transcribe(audio_path, asr_lang)

    if not text or not text.strip():
        raise ValueError("ASR returned empty text")

    if spoken_language == "english":
        src_lang = "eng_Latn"
        tgt_lang = "hin_Deva"
        tts_lang = "hin"
    else:
        src_lang = "hin_Deva"
        tgt_lang = "eng_Latn"
        tts_lang = "eng"

    translated = translator.translate(text, src_lang, tgt_lang)
    try:
        audio_out = tts.synthesize(translated, tts_lang)
    except Exception as e:
        print("⚠️ TTS failed:", e)
        audio_out = None
    
    try:
        save_conversation(spoken_language, text, translated)
    except Exception as e:
        print("⚠️ Conversation save failed:", e)

    try:
        store_voice(
            audio_path=audio_path,
            speaker="aditya",
            language=spoken_language
        )
    except Exception as e:
        print("⚠️ Voice store failed:", e)

    return text, translated, audio_out
