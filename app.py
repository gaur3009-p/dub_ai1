import gradio as gr
from api.routes import process_audio

def pipeline(audio_path, spoken_language):
    if audio_path is None:
        return "", "", None
    return process_audio(audio_path, spoken_language)

with gr.Blocks(title="DubYou – English ⇄ Hindi AI Dubbing") as app:
    gr.Markdown("""
    # 🎙️ DubYou  
    **English ⇄ Hindi AI Dubbing**  
    Whisper → NLLB → MMS-TTS
    """)

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
    app.launch()
