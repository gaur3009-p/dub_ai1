import gradio as gr
from api.routes import process_audio, process_audio_cloned
from services.voice_cloning.enroller import VoiceEnroller
from services.voice_cloning.trainer import VoiceCloningTrainer
from config.voice_clone_settings import CLONE_SUPPORTED_LANGUAGES

enroller = VoiceEnroller()


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 – Standard Dub
# ─────────────────────────────────────────────────────────────────────────────
def standard_pipeline(audio_path, spoken_language):
    if audio_path is None:
        return "", "", None
    return process_audio(audio_path, spoken_language)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 – Voice Enrollment  (Phase 0)
# ─────────────────────────────────────────────────────────────────────────────
def enroll_speaker(speaker_id, audio_files, languages):
    """
    Enroll one or more audio recordings for a speaker.
    audio_files : list of file paths from gr.File(file_count="multiple")
    languages   : list of languages selected via gr.CheckboxGroup
    """
    if not speaker_id or not speaker_id.strip():
        return "❌ Please enter a speaker ID."
    if not audio_files:
        return "❌ Please upload at least one audio file."
    if not languages:
        return "❌ Please select at least one target language."

    paths = [f.name if hasattr(f, "name") else f for f in audio_files]
    try:
        result = enroller.enroll(
            speaker_id=speaker_id.strip(),
            audio_paths=paths,
            languages=languages,
        )
        return (
            f"✅ Speaker **{speaker_id}** enrolled successfully!\n\n"
            f"- Duration: {result['duration_seconds']:.1f}s\n"
            f"- Embedding: {result['embedding_path']}\n"
            f"- Languages queued for training: {', '.join(languages)}"
        )
    except Exception as e:
        return f"❌ Enrollment failed: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 – Clone Training  (Phase 1)
# ─────────────────────────────────────────────────────────────────────────────
def train_voice(speaker_id, audio_files_by_lang, selected_languages, epochs):
    """
    Fine-tune VITS for each selected language.
    Because Gradio doesn't natively support per-language file upload,
    we reuse the same uploaded files for every language (typical short-clip
    scenario where the same speaker speaks multiple langs).
    """
    if not speaker_id or not speaker_id.strip():
        return "❌ Please enter the speaker ID used during enrollment."
    if not audio_files_by_lang:
        return "❌ Please upload training audio files."
    if not selected_languages:
        return "❌ Please select at least one language to train."

    paths = [f.name if hasattr(f, "name") else f for f in audio_files_by_lang]

    try:
        trainer = VoiceCloningTrainer(speaker_id=speaker_id.strip())
    except Exception as e:
        return f"❌ Could not load speaker embedding: {e}\nMake sure you enrolled first."

    log_lines = []
    for lang in selected_languages:
        log_lines.append(f"🔄 Training for **{lang}**…")
        try:
            ckpt = trainer.train(
                language=lang,
                audio_paths=paths,
                epochs=int(epochs),
            )
            log_lines.append(f"  ✅ {lang}: checkpoint saved → `{ckpt}`")
        except Exception as e:
            log_lines.append(f"  ❌ {lang}: {e}")

    return "\n".join(log_lines)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 4 – Cloned Dub
# ─────────────────────────────────────────────────────────────────────────────
def cloned_pipeline(audio_path, spoken_language, target_language, speaker_id):
    if audio_path is None:
        return "", "", None
    if not speaker_id or not speaker_id.strip():
        return "❌ Enter a speaker ID", "", None
    try:
        return process_audio_cloned(
            audio_path=audio_path,
            spoken_language=spoken_language,
            target_language=target_language,
            speaker_id=speaker_id.strip(),
        )
    except Exception as e:
        return str(e), "", None


# ─────────────────────────────────────────────────────────────────────────────
# Build UI
# ─────────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="DubYou – Voice Cloning Dubbing", theme=gr.themes.Soft()) as app:

    gr.Markdown("""
    # 🎙️ DubYou — AI Dubbing with Voice Cloning
    **Whisper ASR → NLLB Translation → Cloned Voice TTS**
    *Supports English · Hindi · French · Spanish*
    """)

    # ── Tab 1 ─────────────────────────────────────────────────────────────────
    with gr.Tab("🔊 Standard Dub (EN ⇄ HI)"):
        gr.Markdown("Quick dub without voice cloning using a generic TTS voice.")
        std_audio  = gr.Audio(sources=["microphone", "upload"], type="filepath", label="🎧 Input Audio")
        std_lang   = gr.Radio(choices=["english", "hindi"], value="english", label="🗣️ Your Language")
        std_btn    = gr.Button("🚀 Translate & Dub", variant="primary")
        std_recog  = gr.Textbox(label="📝 Recognized Speech")
        std_trans  = gr.Textbox(label="🌍 Translated Text")
        std_out    = gr.Audio(label="🔊 Dubbed Audio", autoplay=True)
        std_btn.click(standard_pipeline, [std_audio, std_lang], [std_recog, std_trans, std_out])

    # ── Tab 2 ─────────────────────────────────────────────────────────────────
    with gr.Tab("👤 Phase 0 · Enroll Voice"):
        gr.Markdown("""
        ### Step 1 — Enroll your voice
        Upload **10+ seconds** of clear speech (WAV/MP3).
        Multiple files are averaged for a better embedding.
        """)
        enr_id      = gr.Textbox(label="Speaker ID (e.g. your name)", placeholder="aditya")
        enr_files   = gr.File(label="🎤 Upload Audio Samples", file_count="multiple",
                               file_types=[".wav", ".mp3", ".flac", ".ogg"])
        enr_langs   = gr.CheckboxGroup(
            choices=CLONE_SUPPORTED_LANGUAGES,
            value=CLONE_SUPPORTED_LANGUAGES,
            label="🌐 Languages to clone voice into",
        )
        enr_btn     = gr.Button("✅ Enroll Speaker", variant="primary")
        enr_status  = gr.Markdown()
        enr_btn.click(enroll_speaker, [enr_id, enr_files, enr_langs], enr_status)

    # ── Tab 3 ─────────────────────────────────────────────────────────────────
    with gr.Tab("🏋️ Phase 1 · Train Clone"):
        gr.Markdown("""
        ### Step 2 — Fine-tune for each language
        Upload audio **with matching .txt transcript files** alongside each WAV.
        Training uses the enrolled speaker embedding as a conditioning signal.
        > ⏱️ Expect ~5–30 min per language on GPU; longer on CPU.
        """)
        trn_id      = gr.Textbox(label="Speaker ID (same as enrollment)", placeholder="aditya")
        trn_files   = gr.File(label="🎤 Upload Training Audio (+ .txt sidecars)",
                               file_count="multiple",
                               file_types=[".wav", ".mp3", ".flac"])
        trn_langs   = gr.CheckboxGroup(
            choices=CLONE_SUPPORTED_LANGUAGES,
            value=["english", "hindi"],
            label="🌐 Languages to train",
        )
        trn_epochs  = gr.Slider(minimum=10, maximum=500, value=200, step=10, label="Epochs")
        trn_btn     = gr.Button("🚀 Start Training", variant="primary")
        trn_log     = gr.Markdown()
        trn_btn.click(train_voice, [trn_id, trn_files, trn_langs, trn_epochs], trn_log)

    # ── Tab 4 ─────────────────────────────────────────────────────────────────
    with gr.Tab("🎭 Cloned Dub"):
        gr.Markdown("""
        ### Step 3 — Dub with your cloned voice
        Speak in any supported language; hear yourself dubbed in another.
        """)
        cln_audio  = gr.Audio(sources=["microphone", "upload"], type="filepath", label="🎧 Input Audio")
        with gr.Row():
            cln_src  = gr.Dropdown(choices=CLONE_SUPPORTED_LANGUAGES, value="english", label="🗣️ Spoken Language")
            cln_tgt  = gr.Dropdown(choices=CLONE_SUPPORTED_LANGUAGES, value="hindi",   label="🎯 Target Language")
        cln_spk    = gr.Textbox(label="👤 Speaker ID", placeholder="aditya")
        cln_btn    = gr.Button("🎭 Clone & Dub", variant="primary")
        cln_recog  = gr.Textbox(label="📝 Recognized Speech")
        cln_trans  = gr.Textbox(label="🌍 Translated Text")
        cln_out    = gr.Audio(label="🔊 Cloned Dubbed Audio", autoplay=True)
        cln_btn.click(
            cloned_pipeline,
            [cln_audio, cln_src, cln_tgt, cln_spk],
            [cln_recog, cln_trans, cln_out],
        )

    gr.Markdown("---\n*DubYou · Powered by Whisper · NLLB · MMS-TTS · YourTTS · SpeechBrain*")


if __name__ == "__main__":
    app.launch(share=True)
