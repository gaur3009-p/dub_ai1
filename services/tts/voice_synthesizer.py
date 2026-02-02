import torch
import soundfile as sf
import uuid
from transformers import VitsModel, AutoTokenizer
from config.settings import MMS_TTS_BASE

class VoiceSynthesizer:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}

    def _load_model(self, lang: str):
        model_name = f"{MMS_TTS_BASE}-{lang}"

        if lang not in self.models:
            self.tokenizers[lang] = AutoTokenizer.from_pretrained(model_name)
            self.models[lang] = VitsModel.from_pretrained(model_name)

        return self.models[lang], self.tokenizers[lang]

    def synthesize(self, text: str, lang: str = "eng") -> str:
        model, tokenizer = self._load_model(lang)

        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            waveform = model(**inputs).waveform

        output_path = f"/tmp/{uuid.uuid4()}.wav"
        sf.write(output_path, waveform.squeeze().cpu().numpy(), 16000)

        return output_path
