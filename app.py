import gradio as gr
from api.routes import process_audio, enroll_voice_http

def enroll(audio_path):
    enroll_voice_http(audio_path, speaker="aditya")
    return "✅ Voice enrolled successfully"

def pipeline(audio_path, spoken_language):
    if not audio_path:
        return "", "", None
    return process_audio(audio_path, spoken_language)

with gr.Blocks(title="DubYou") as app:
    gr.Markdown("# 🎙️ DubYou – AI Voice Cloning (Microservice)")

    gr.Markdown("## 🧬 Phase 0: Voice Enrollment")
    enroll_audio = gr.Audio(type="filepath")
    enroll_btn = gr.Button("Enroll Voice")
    enroll_status = gr.Textbox()

    enroll_btn.click(enroll, enroll_audio, enroll_status)

    gr.Markdown("---")
    gr.Markdown("## 🎧 Translate & Dub")

    audio = gr.Audio(type="filepath")
    lang = gr.Radio(["english", "hindi"], value="english")

    run = gr.Button("Translate & Dub")
    out1 = gr.Textbox(label="Recognized")
    out2 = gr.Textbox(label="Translated")
    out3 = gr.Audio(label="Dubbed", autoplay=True)

    run.click(pipeline, [audio, lang], [out1, out2, out3])

if __name__ == "__main__":
    app.launch(share=True)
