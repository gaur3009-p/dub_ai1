from services.asr.whisper_asr import WhisperASR
from services.translation.nllb_translator import NLLBTranslator
from services.tts.voice_synthesizer import VoiceSynthesizer
from config.settings import NLLB_MODEL

asr = WhisperASR()
translator = NLLBTranslator(NLLB_MODEL)
tts = VoiceSynthesizer()

def process_audio(audio_path, src_lang, tgt_lang, tts_lang):
    text = asr.transcribe(audio_path)
    translated = translator.translate(text, src_lang, tgt_lang)
    output_audio = tts.synthesize(translated, tts_lang)
    return text, translated, output_audio
