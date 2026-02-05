import uuid
import re
import numpy as np
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

        # Allow English + Devanagari + basic punctuation
        text = re.sub(r"[^\w\s\u0900-\u097F.,?!]", "", text)

        # Collapse multiple spaces
        text = re.sub(r"\s+", " ", text)

        return text

    def synthesize(self, text: str, lang: str) -> str:
        if lang not in MMS_TTS_LANGUAGES.values():
            raise ValueError(f"Unsupported TTS language: {lang}")

        text = self._sanitize_text(text)

        # 🚨 MMS-TTS cannot handle very short / fragment text
        if len(text) < 3:
            raise ValueError("TTS text too short after sanitization")

        tts_pipe = self._load_pipe(lang)
        output = tts_pipe(text)

        audio = output["audio"]
        sr = output["sampling_rate"]

        # 🔒 Ensure numpy float32 for soundfile
        if not isinstance(audio, np.ndarray):
            audio = np.array(audio, dtype=np.float32)
        else:
            audio = audio.astype(np.float32)

        output_path = f"/tmp/{uuid.uuid4()}.wav"

        # 🔥 Explicit WAV format to avoid corruption
        sf.write(
            output_path,
            audio,
            sr,
            format="WAV",
            subtype="PCM_16"
        )

        return output_path
