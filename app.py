import gradio as gr
from api.routes import process_audio

def pipeline(audio_path, src_lang, tgt_lang, tts_lang):
    if audio_path is None:
        return "", "", None

    recognized_text, translated_text, dubbed_audio = process_audio(
        audio_path=audio_path,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        tts_lang=tts_lang
    )

    return recognized_text, translated_text, dubbed_audio


with gr.Blocks(title="DubYou – Multilingual AI Dubbing") as app:
    gr.Markdown(
        """
        # 🎙️ DubYou  
        **Real-time Multilingual AI Dubbing (Python 3.12 Safe)**  
        Whisper → NLLB → MMS-TTS
        """
    )

    with gr.Row():
        audio_input = gr.Audio(
            sources=["microphone", "upload"],
            type="filepath",
            label="🎧 Speak or Upload Audio"
        )

    with gr.Row():
        src_lang = gr.Textbox(
            value="eng_Latn",
            label="ASR / Source Language (NLLB code)"
        )
        tgt_lang = gr.Textbox(
            value="hin_Deva",
            label="Translation Target Language (NLLB code)"
        )
        tts_lang = gr.Textbox(
            value="hin",
            label="TTS Language (MMS code)"
        )

    run_btn = gr.Button("🚀 Translate & Dub", variant="primary")

    with gr.Row():
        recognized_text = gr.Textbox(
            label="📝 Recognized Speech",
            lines=3
        )
        translated_text = gr.Textbox(
            label="🌍 Translated Text",
            lines=3
        )

    dubbed_audio = gr.Audio(
        label="🔊 Dubbed Output Voice",
        autoplay=True
    )

    run_btn.click(
        fn=pipeline,
        inputs=[audio_input, src_lang, tgt_lang, tts_lang],
        outputs=[recognized_text, translated_text, dubbed_audio]
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
