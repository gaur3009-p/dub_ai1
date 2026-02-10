from fastapi import FastAPI, UploadFile
import torch, tempfile
import soundfile as sf
from TTS.tts.models.xtts import Xtts
from TTS.tts.configs.xtts_config import XttsConfig

app = FastAPI()

config = XttsConfig()
config.load_json("tts_models/multilingual/multi-dataset/xtts_v2/config.json")

model = Xtts.init_from_config(config)
model.load_checkpoint(
    config,
    checkpoint_path="tts_models/multilingual/multi-dataset/xtts_v2/model.pth",
    vocab_path="tts_models/multilingual/multi-dataset/xtts_v2/vocab.json",
    speaker_file_path="tts_models/multilingual/multi-dataset/xtts_v2/speakers_xtts.pth",
)

model.cuda()
model.eval()

SPEAKERS = {}

@app.post("/enroll")
async def enroll(speaker: str, audio: UploadFile):
    wav, _ = sf.read(audio.file)
    wav = torch.tensor(wav).unsqueeze(0).cuda()
    SPEAKERS[speaker] = model.get_speaker_embedding(wav)
    return {"status": "enrolled"}

@app.post("/synthesize")
async def synthesize(speaker: str, text: str, language: str):
    emb = SPEAKERS.get(speaker)
    if emb is None:
        return {"error": "Speaker not enrolled"}

    with torch.no_grad():
        audio = model.inference(
            text=text,
            language=language,
            speaker_embedding=emb
        )

    f = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(f.name, audio, 24000)
    return {"audio_path": f.name}
