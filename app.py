import gradio as gr
import requests
from services.pipeline import process_audio

VOICE_URL = "http://localhost:8001"

def enroll(audio_path):
    with open(audio_path, "rb") as f:
        requests.post(
            f"{VOICE_URL}/enroll",
            data={"speaker": "aditya"},
            files={"audio": f}
        )
    return "Voice enrolled"

def run(audio_path, lang):

    text, translated = process_audio(audio_path, lang)

    r = requests.post(
        f"{VOICE_URL}/synthesize",
        data={
            "speaker": "aditya",
            "text": translated,
            "language": "hi"
        }
    )

    return text, translated, r.json()["audio_path"]


with gr.Blocks() as app:

    gr.Markdown("# DubYou Real-Time Translator")

    enroll_audio = gr.Audio(type="filepath")
    enroll_btn = gr.Button("Enroll")
    enroll_status = gr.Textbox()

    enroll_btn.click(enroll, enroll_audio, enroll_status)

    gr.Markdown("---")

    audio = gr.Audio(type="filepath")
    lang = gr.Dropdown(list({
        "english","hindi","bengali","tamil",
        "telugu","marathi","gujarati"
    }))

    btn = gr.Button("Translate")

    t1 = gr.Textbox(label="Recognized")
    t2 = gr.Textbox(label="Translated")
    out = gr.Audio(label="Dubbed")

    btn.click(run, [audio, lang], [t1, t2, out])

app.launch()
