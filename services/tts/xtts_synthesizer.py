import uuid
import torch
import soundfile as sf
from transformers import AutoModel, AutoProcessor

class XTTSVoiceCloner:
    def __init__(self):
        self.processor = AutoProcessor.from_pretrained("coqui/XTTS-v2")
        self.model = AutoModel.from_pretrained(
            "coqui/XTTS-v2",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )

    def synthesize(self, text: str, speaker_embedding, language: str) -> str:
        with torch.no_grad():
            audio = self.model.generate(
                text=text,
                speaker_embedding=speaker_embedding,
                language=language
            )

        output_path = f"/tmp/{uuid.uuid4()}.wav"
        sf.write(output_path, audio, 24000)
        return output_path
