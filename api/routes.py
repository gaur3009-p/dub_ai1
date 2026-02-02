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
    """
    spoken_language: 'english' or 'hindi'
    """

    # 1️⃣ ASR
    asr_lang = SUPPORTED_ASR_LANGS[spoken_language]
    text = asr.transcribe(audio_path, asr_lang)

    # 2️⃣ Translation direction
    if spoken_language == "english":
        src_lang = NLLB_LANG_MAP["english"]
        tgt_lang = NLLB_LANG_MAP["hindi"]
        tts_lang = MMS_TTS_LANGUAGES["hindi"]
    else:
        src_lang = NLLB_LANG_MAP["hindi"]
        tgt_lang = NLLB_LANG_MAP["english"]
        tts_lang = MMS_TTS_LANGUAGES["english"]

    translated = translator.translate(text, src_lang, tgt_lang)

    # 3️⃣ TTS
    output_audio = tts.synthesize(translated, tts_lang)

    return text, translated, output_audio
