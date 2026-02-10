from fastapi import FastAPI, UploadFile
import tempfile
import soundfile as sf
import torch
from TTS.tts.models.xtts import Xtts
from TTS.tts.configs.xtts_config import XttsConfig

app = FastAPI()

# -----------------------------
# Load XTTS once (GPU)
# -----------------------------
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

SPEAKERS = {}  # in-memory speaker store


@app.post("/enroll")
async def enroll(speaker: str, audio: UploadFile):
    wav, sr = sf.read(audio.file)
    wav = torch.tensor(wav).unsqueeze(0).cuda()

    emb = model.get_speaker_embedding(wav)
    SPEAKERS[speaker] = emb

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

    out = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(out.name, audio, 24000)

    return {"audio_path": out.name}
