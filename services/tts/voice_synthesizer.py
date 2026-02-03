import uuid
import re
import soundfile as sf
from transformers import pipeline
from config.settings import MMS_TTS_BASE, MMS_TTS_LANGUAGES


class VoiceSynthesizer:
    def __init__(self):
        self.pipes = {}

    def _load_pipe(self, lang: str):
        if lang not in self.pipes:
            model_name = f"{MMS_TTS_BASE}-{lang}"
            self.pipes[lang] = pipeline(
                task="text-to-speech",
                model=model_name,
            )
        return self.pipes[lang]

    def _sanitize_text(self, text: str) -> str:
        text = text.strip()

        # Remove non-speakable characters
        text = re.sub(r"[^\w\s\u0900-\u097F.,?!]", "", text)

        # Collapse spaces
        text = re.sub(r"\s+", " ", text)

        return text

    def synthesize(self, text: str, lang: str) -> str:
        if lang not in MMS_TTS_LANGUAGES.values():
            raise ValueError(f"Unsupported TTS language: {lang}")

        text = self._sanitize_text(text)

        # 🚨 CRITICAL: MMS cannot handle very short text
        if len(text) < 3:
            raise ValueError("TTS text too short after sanitization")

        tts_pipe = self._load_pipe(lang)
        output = tts_pipe(text)

        output_path = f"/tmp/{uuid.uuid4()}.wav"
        sf.write(
            output_path,
            output["audio"],
            output["sampling_rate"]
        )

        return output_path
