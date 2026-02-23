import base64
import io
import os
import tempfile
import traceback

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from TTS.api import TTS

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="YourTTS Microservice", version="1.0.0")

# ── Supported language codes for YourTTS ─────────────────────────────────────
LANG_MAP = {
    "english": "en",
    "hindi":   "hi",
    "french":  "fr-fr",
    "spanish": "es",
}

# ── Lazy-load model ───────────────────────────────────────────────────────────
_tts: TTS | None = None


def get_tts() -> TTS:
    global _tts
    if _tts is None:
        print("[YourTTS] Loading model…")
        _tts = TTS(
            model_name="tts_models/multilingual/multi-dataset/your_tts",
            progress_bar=False,
            gpu=os.environ.get("USE_GPU", "0") == "1",
        )
        print("[YourTTS] ✅ Model ready.")
    return _tts


# ── Request / Response schemas ────────────────────────────────────────────────
class SynthesizeRequest(BaseModel):
    text: str
    language: str                   # "english" | "hindi" | "french" | "spanish"
    speaker_wav_b64: str | None = None   # base64-encoded reference WAV


class SynthesizeResponse(BaseModel):
    audio_b64: str                  # base64-encoded WAV
    sample_rate: int


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": "your_tts"}


@app.post("/synthesize", response_model=SynthesizeResponse)
def synthesize(req: SynthesizeRequest):
    if req.language not in LANG_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language '{req.language}'. "
                   f"Supported: {list(LANG_MAP.keys())}",
        )

    tts = get_tts()
    lang_code = LANG_MAP[req.language]

    # Write reference audio to a temp file if provided
    ref_wav_path: str | None = None
    tmp_ref = None
    if req.speaker_wav_b64:
        audio_bytes = base64.b64decode(req.speaker_wav_b64)
        tmp_ref = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_ref.write(audio_bytes)
        tmp_ref.flush()
        ref_wav_path = tmp_ref.name

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
            out_path = tmp_out.name

        tts.tts_to_file(
            text=req.text,
            language=lang_code,
            speaker_wav=ref_wav_path,
            file_path=out_path,
        )

        # Read output and encode
        audio_data, sr = sf.read(out_path)
        audio_data = audio_data.astype(np.float32)

        buf = io.BytesIO()
        sf.write(buf, audio_data, sr, format="WAV", subtype="PCM_16")
        audio_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return SynthesizeResponse(audio_b64=audio_b64, sample_rate=sr)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup temp files
        for path in [ref_wav_path, out_path if "out_path" in dir() else None]:
            if path and os.path.exists(path):
                os.unlink(path)
        if tmp_ref:
            tmp_ref.close()


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=False)
