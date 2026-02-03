import re
import torch
import soundfile as sf
import uuid
from transformers import VitsModel, AutoTokenizer
from config.settings import MMS_TTS_BASE, MMS_TTS_LANGUAGES

class VoiceSynthesizer:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}

    def _load(self, lang: str):
        if lang not in self.models:
            model_name = f"{MMS_TTS_BASE}-{lang}"
            self.tokenizers[lang] = AutoTokenizer.from_pretrained(model_name)
            self.models[lang] = VitsModel.from_pretrained(model_name)
        return self.models[lang], self.tokenizers[lang]

    def _sanitize_text(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r"\s+", " ", text)  # normalize spaces
        return text

    def synthesize(self, text: str, lang: str) -> str:
        if lang not in MMS_TTS_LANGUAGES.values():
            raise ValueError(f"Unsupported TTS language: {lang}")

        text = self._sanitize_text(text)

        if not text:
            raise ValueError("TTS received empty text")

        model, tokenizer = self._load(lang)

        inputs = tokenizer(text, return_tensors="pt")
        if inputs["input_ids"].shape[1] == 0:
            raise ValueError("Tokenizer produced empty input_ids")

        with torch.no_grad():
            outputs = model(**inputs)
            waveform = outputs.waveform

        output_path = f"/tmp/{uuid.uuid4()}.wav"
        sf.write(output_path, waveform.squeeze().cpu().numpy(), 16000)

        return output_path
