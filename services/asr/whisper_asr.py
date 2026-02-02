import whisper
from config.settings import WHISPER_MODEL

class WhisperASR:
    def __init__(self):
        self.model = whisper.load_model(WHISPER_MODEL)

    def transcribe(self, audio_path: str) -> str:
        result = self.model.transcribe(audio_path)
        return result["text"]
