from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
import torch
import tempfile
import soundfile as sf
from TTS.api import TTS

app = FastAPI()

# -----------------------------
# Load XTTS v2 properly
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

print("🚀 Loading XTTS v2...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
print("✅ XTTS Loaded Successfully")

# Store speaker embeddings in memory
SPEAKERS = {}


# -----------------------------
# Voice Enrollment
# -----------------------------
@app.post("/enroll")
async def enroll(
    speaker: str = Form(...),
    audio: UploadFile = None
):
    if audio is None:
        return JSONResponse({"error": "No audio file provided"}, status_code=400)

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await audio.read())
        temp_path = tmp.name

    # Extract speaker embedding
    try:
        embedding = tts.get_speaker_embedding(temp_path)
        SPEAKERS[speaker] = embedding
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    return {"status": f"Speaker '{speaker}' enrolled successfully"}


# -----------------------------
# Voice Cloning / Synthesis
# -----------------------------
@app.post("/synthesize")
async def synthesize(
    speaker: str = Form(...),
    text: str = Form(...),
    language: str = Form(...)
):
    emb = SPEAKERS.get(speaker)

    if emb is None:
        return JSONResponse(
            {"error": "Speaker not enrolled"},
            status_code=400
        )

    # Generate audio file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        output_path = tmp.name

    try:
        tts.tts_to_file(
            text=text,
            speaker_embedding=emb,
            language=language,
            file_path=output_path
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    return {"audio_path": output_path}
