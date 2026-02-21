import os
os.environ["COQUI_TOS_AGREED"] = "1"

import torch
import asyncio
from fastapi import FastAPI
from aiortc import RTCPeerConnection, RTCSessionDescription
from TTS.api import TTS
import whisper
import tempfile

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load models once
whisper_model = whisper.load_model("base")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

SPEAKERS = {}
pcs = set()

@app.post("/offer")
async def offer(request: dict):
    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("track")
    async def on_track(track):
        if track.kind == "audio":
            while True:
                frame = await track.recv()
                # Convert frame to wav chunk
                with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
                    tmp.write(frame.to_bytes())
                    tmp.flush()

                    # Streaming ASR
                    result = whisper_model.transcribe(tmp.name)
                    text = result["text"]

                    # Translate here (NLLB call)
                    translated = text  # placeholder

                    # TTS
                    output_path = "/tmp/out.wav"
                    tts.tts_to_file(
                        text=translated,
                        speaker_embedding=SPEAKERS["aditya"],
                        language="hi",
                        file_path=output_path
                    )

    await pc.setRemoteDescription(
        RTCSessionDescription(sdp=request["sdp"], type=request["type"])
    )

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    }
