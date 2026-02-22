from services.asr.whisper_asr import WhisperASR
from services.translation.nllb_translator import NLLBTranslator
from services.memory.qdrant_memory import MemoryStore
from config.settings import SUPPORTED_ASR_LANGS, NLLB_LANG_MAP

asr = WhisperASR()
translator = NLLBTranslator("facebook/nllb-200-distilled-600M")
memory = MemoryStore()


def process_audio(audio_path, spoken_language):

    # ASR
    text = asr.transcribe(
        audio_path,
        SUPPORTED_ASR_LANGS[spoken_language]
    )

    if not text:
        raise ValueError("Empty speech")

    # Translate
    src_lang = NLLB_LANG_MAP[spoken_language]
    tgt_lang = "eng_Latn" if spoken_language != "english" else "hin_Deva"

    translated = translator.translate(text, src_lang, tgt_lang)

    # Save memory async style (non-blocking optional)
    memory.save(text, translated)

    return text, translated
