import uuid
import torch
import soundfile as sf
from TTS.tts.models.xtts import Xtts
from TTS.tts.configs.xtts_config import XttsConfig


class XTTSVoiceCloner:
    def __init__(self):
        config = XttsConfig()
        config.load_json("tts_models/multilingual/multi-dataset/xtts_v2/config.json")

        self.model = Xtts.init_from_config(config)
        self.model.load_checkpoint(
            config,
            checkpoint_path="tts_models/multilingual/multi-dataset/xtts_v2/model.pth",
            vocab_path="tts_models/multilingual/multi-dataset/xtts_v2/vocab.json",
            speaker_file_path="tts_models/multilingual/multi-dataset/xtts_v2/speakers_xtts.pth",
        )

        self.model.cuda() if torch.cuda.is_available() else self.model.cpu()
        self.model.eval()

    def synthesize(self, text: str, speaker_embedding, language: str) -> str:
        with torch.no_grad():
            audio = self.model.inference(
                text=text,
                language=language,
                speaker_embedding=torch.tensor(speaker_embedding).unsqueeze(0)
            )

        output_path = f"/tmp/{uuid.uuid4()}.wav"
        sf.write(output_path, audio, 24000)
        return output_path
