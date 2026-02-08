import gradio as gr
from api.routes import process_audio
from services.voice_identity.voice_storage import store_voice


# =====================================================
# Phase 0: Voice Enrollment (One-time)
# =====================================================
def enroll_voice(audio_path):
    if audio_path is None:
        return "❌ Please record or upload audio first."

    try:
        store_voice(
            audio_path=audio_path,
            speaker="aditya",
            language="en"
        )
        return "✅ Voice enrolled successfully. You can now dub in other languages."
    except Exception as e:
        return f"❌ Voice enrollment failed: {e}"


# =====================================================
# Phase 2: Translate & Dub
# =====================================================
def pipeline(audio_path, spoken_language):
    if audio_path is None:
        return "", "", None
    return process_audio(audio_path, spoken_language)


# =====================================================
# UI
# =====================================================
with gr.Blocks(title="DubYou – Multilingual Voice Cloning") as app:
    gr.Markdown("""
    # 🎙️ DubYou  
    **Multilingual AI Voice Cloning & Dubbing**

    **Pipeline:**  
    Voice Enrollment → Whisper ASR → NLLB Translation → Voice Cloning TTS
    """)

    # -------------------------------
    # Phase 0: Voice Enrollment
    # -------------------------------
    gr.Markdown("## 🧬 Phase 0: Voice Enrollment (One Time Setup)")

    enroll_audio = gr.Audio(
        sources=["microphone", "upload"],
        type="filepath",
        label="🎤 Record or Upload Your Voice (5–15 minutes recommended)"
    )

    enroll_btn = gr.Button("🧬 Enroll Voice", variant="primary")
    enroll_status = gr.Textbox(
        label="Enrollment Status",
        interactive=False
    )

    enroll_btn.click(
        enroll_voice,
        inputs=[enroll_audio],
        outputs=[enroll_status]
    )

    gr.Markdown("---")

    # -------------------------------
    # Phase 2: Translate & Dub
    # -------------------------------
    gr.Markdown("## 🎧 Phase 2: Translate & Dub")

    audio = gr.Audio(
        sources=["microphone", "upload"],
        type="filepath",
        label="🎧 Speak or Upload Audio"
    )

    spoken_language = gr.Radio(
        choices=["english", "hindi"],
        value="english",
        label="🗣️ Language You Are Speaking"
    )

    run_btn = gr.Button("🚀 Translate & Dub", variant="primary")

    recognized = gr.Textbox(label="📝 Recognized Speech")
    translated = gr.Textbox(label="🌍 Translated Text")
    dubbed = gr.Audio(label="🔊 Dubbed Voice", autoplay=True)

    run_btn.click(
        pipeline,
        inputs=[audio, spoken_language],
        outputs=[recognized, translated, dubbed]
    )


if __name__ == "__main__":
    app.launch(share=True)
