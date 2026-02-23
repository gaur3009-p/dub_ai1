from services.asr.whisper_asr import WhisperASR
from services.translation.nllb_translator import NLLBTranslator
from services.tts.voice_synthesizer import VoiceSynthesizer
from services.voice_cloning.cloned_synthesizer import ClonedVoiceSynthesizer
from config.settings import (
    SUPPORTED_ASR_LANGS,
    NLLB_LANG_MAP,
    MMS_TTS_LANGUAGES,
    NLLB_MODEL,
)
from config.voice_clone_settings import CLONE_LANG_MAP, CLONE_SUPPORTED_LANGUAGES
from services.voice_identity.voice_storage import store_voice
from services.database.conversation_repo import save_conversation

# ── Singletons ────────────────────────────────────────────────────────────────
asr              = WhisperASR()
translator       = NLLBTranslator(NLLB_MODEL)
tts              = VoiceSynthesizer()          # original MMS-TTS
cloned_synth     = ClonedVoiceSynthesizer()    # fine-tuned / YourTTS


# ── Helper ────────────────────────────────────────────────────────────────────
def _get_nllb_pair(spoken_language: str, target_language: str):
    """Return (src_lang, tgt_lang) NLLB codes."""
    src = CLONE_LANG_MAP[spoken_language]["nllb_src"]
    tgt = CLONE_LANG_MAP[target_language]["nllb_tgt"]
    return src, tgt


# ── Original pipeline (unchanged) ────────────────────────────────────────────
def process_audio(audio_path: str, spoken_language: str):
    """
    Legacy pipeline: English ⇄ Hindi only, generic MMS-TTS voice.
    """
    asr_lang = SUPPORTED_ASR_LANGS[spoken_language]
    text = asr.transcribe(audio_path, asr_lang)
    if not text or not text.strip():
        raise ValueError("ASR returned empty text")

    if spoken_language == "english":
        src_lang, tgt_lang, tts_lang = "eng_Latn", "hin_Deva", "hin"
    else:
        src_lang, tgt_lang, tts_lang = "hin_Deva", "eng_Latn", "eng"

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
        store_voice(audio_path=audio_path, speaker="aditya", language=spoken_language)
    except Exception as e:
        print("⚠️ Voice store failed:", e)

    return text, translated, audio_out


# ── Cloned-voice pipeline (new) ───────────────────────────────────────────────
def process_audio_cloned(
    audio_path: str,
    spoken_language: str,
    target_language: str,
    speaker_id: str,
):
    """
    Cloned pipeline: any → any language, using the speaker's cloned voice.

    Parameters
    ----------
    audio_path       : path to uploaded / recorded audio
    spoken_language  : language the user is speaking in
    target_language  : language to dub into
    speaker_id       : enrolled speaker whose voice to clone
    """
    if spoken_language not in CLONE_SUPPORTED_LANGUAGES:
        raise ValueError(f"Spoken language '{spoken_language}' not supported.")
    if target_language not in CLONE_SUPPORTED_LANGUAGES:
        raise ValueError(f"Target language '{target_language}' not supported.")

    # ── Step 1: ASR ───────────────────────────────────────────────────────────
    asr_lang = CLONE_LANG_MAP[spoken_language]["whisper"]
    text = asr.transcribe(audio_path, asr_lang)
    if not text or not text.strip():
        raise ValueError("ASR returned empty text.")

    # ── Step 2: Translate ─────────────────────────────────────────────────────
    if spoken_language == target_language:
        translated = text                   # no translation needed
    else:
        src_lang, tgt_lang = _get_nllb_pair(spoken_language, target_language)
        translated = translator.translate(text, src_lang, tgt_lang)

    # ── Step 3: Cloned TTS ────────────────────────────────────────────────────
    audio_out = None
    try:
        audio_out = cloned_synth.synthesize(
            text=translated,
            language=target_language,
            speaker_id=speaker_id,
            reference_audio=audio_path,
        )
    except Exception as e:
        print(f"⚠️ Cloned TTS failed: {e}")

    # ── Step 4: Persist ───────────────────────────────────────────────────────
    try:
        save_conversation(spoken_language, text, translated)
    except Exception as e:
        print(f"⚠️ Conversation save failed: {e}")

    try:
        store_voice(audio_path=audio_path, speaker=speaker_id, language=spoken_language)
    except Exception as e:
        print(f"⚠️ Voice store failed: {e}")

    return text, translated, audio_out
