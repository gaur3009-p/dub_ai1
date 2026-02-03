from services.asr.whisper_asr import WhisperASR
from services.translation.nllb_translator import NLLBTranslator
from services.tts.voice_synthesizer import VoiceSynthesizer
from config.settings import (
    SUPPORTED_ASR_LANGS,
    NLLB_LANG_MAP,
    MMS_TTS_LANGUAGES,
    NLLB_MODEL
)

asr = WhisperASR()
translator = NLLBTranslator(NLLB_MODEL)
tts = VoiceSynthesizer()

def process_audio(audio_path: str, spoken_language: str):
    asr_lang = SUPPORTED_ASR_LANGS[spoken_language]
    text = asr.transcribe(audio_path, asr_lang)

    if spoken_language == "english":
        src, tgt = NLLB_LANG_MAP["english"], NLLB_LANG_MAP["hindi"]
        tts_lang = MMS_TTS_LANGUAGES["hindi"]
    else:
        src, tgt = NLLB_LANG_MAP["hindi"], NLLB_LANG_MAP["english"]
        tts_lang = MMS_TTS_LANGUAGES["english"]

    translated = translator.translate(text, src, tgt)
    audio_out = tts.synthesize(translated, tts_lang)

    return text, translated, audio_out
