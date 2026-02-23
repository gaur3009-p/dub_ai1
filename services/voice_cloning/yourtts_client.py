import base64
import os
import tempfile

import httpx
import soundfile as sf

YOURTTS_SERVICE_URL = os.getenv("YOURTTS_SERVICE_URL", "http://localhost:8001")
_TIMEOUT = 120.0   # seconds – TTS can be slow on CPU


class YourTTSClient:
    """Thin HTTP wrapper around the YourTTS microservice."""

    def __init__(self, base_url: str = YOURTTS_SERVICE_URL):
        self.base_url = base_url.rstrip("/")

    def health(self) -> bool:
        try:
            r = httpx.get(f"{self.base_url}/health", timeout=5.0)
            return r.status_code == 200
        except Exception:
            return False

    def synthesize(
        self,
        text: str,
        language: str,
        speaker_wav_path: str | None = None,
    ) -> str:
        """
        Call the YourTTS service and return path to synthesized WAV.

        Parameters
        ----------
        text             : text to synthesize
        language         : "english" | "hindi" | "french" | "spanish"
        speaker_wav_path : path to reference audio for voice cloning

        Returns
        -------
        Path to a temporary WAV file.
        """
        # Encode reference audio if provided
        speaker_wav_b64 = None
        if speaker_wav_path and os.path.exists(speaker_wav_path):
            with open(speaker_wav_path, "rb") as f:
                speaker_wav_b64 = base64.b64encode(f.read()).decode("utf-8")

        payload = {
            "text": text,
            "language": language,
            "speaker_wav_b64": speaker_wav_b64,
        }

        try:
            response = httpx.post(
                f"{self.base_url}/synthesize",
                json=payload,
                timeout=_TIMEOUT,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"YourTTS service error {e.response.status_code}: "
                f"{e.response.text}"
            ) from e
        except httpx.RequestError as e:
            raise RuntimeError(
                f"Cannot reach YourTTS service at {self.base_url}: {e}"
            ) from e

        data = response.json()
        audio_bytes = base64.b64decode(data["audio_b64"])
        sr = data["sample_rate"]

        # Write to temp file
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(audio_bytes)
        tmp.flush()
        tmp.close()
        return tmp.name
