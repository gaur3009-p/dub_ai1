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
    # 1️⃣ ASR
    asr_lang = SUPPORTED_ASR_LANGS[spoken_language]
    text = asr.transcribe(audio_path, asr_lang)

    if not text or not text.strip():
        raise ValueError("ASR returned empty text")

    # 2️⃣ Decide translation direction
    if spoken_language == "english":
        src_lang = NLLB_LANG_MAP["english"]
        tgt_lang = NLLB_LANG_MAP["hindi"]
        tts_lang = MMS_TTS_LANGUAGES["hindi"]
    else:
        src_lang = NLLB_LANG_MAP["hindi"]
        tgt_lang = NLLB_LANG_MAP["english"]
        tts_lang = MMS_TTS_LANGUAGES["english"]

    translated = translator.translate(text, src_lang, tgt_lang)
    audio_out = tts.synthesize(translated, tts_lang)

    save_conversation(spoken_language, text, translated)
    store_voice(
        audio_path=audio_path,
        speaker="aditya",      
        language=spoken_language
    )

    return text, translated, audio_out
