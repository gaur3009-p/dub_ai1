import uuid
import os
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Optional

from transformers import VitsModel, VitsTokenizer
from config.voice_clone_settings import (
    CLONE_CHECKPOINTS_DIR,
    YOURTTS_MODEL,
    CLONE_SUPPORTED_LANGUAGES,
)
from services.voice_cloning.enroller import VoiceEnroller


class ClonedVoiceSynthesizer:
    """Synthesize speech in a cloned voice for any supported language."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._vits_cache: dict = {}          # (speaker, lang) → (model, tokenizer)
        self._yourtts_pipe = None            # lazy-loaded

    # ── VITS (fine-tuned) ─────────────────────────────────────────────────────
    def _ckpt_path(self, speaker_id: str, language: str) -> Optional[Path]:
        p = Path(CLONE_CHECKPOINTS_DIR) / speaker_id / language
        if p.exists() and (p / "config.json").exists():
            return p
        return None

    def _load_vits(self, speaker_id: str, language: str):
        key = (speaker_id, language)
        if key not in self._vits_cache:
            ckpt = self._ckpt_path(speaker_id, language)
            if ckpt is None:
                raise FileNotFoundError(
                    f"No fine-tuned checkpoint for speaker='{speaker_id}' lang='{language}'. "
                    "Run VoiceCloningTrainer.train() first."
                )
            tokenizer = VitsTokenizer.from_pretrained(str(ckpt))
            model = VitsModel.from_pretrained(str(ckpt)).to(self.device)
            model.eval()
            self._vits_cache[key] = (model, tokenizer)
            print(f"[ClonedSynth] ✅ Loaded VITS for {speaker_id}/{language}")
        return self._vits_cache[key]

    def _synthesize_vits(
        self,
        text: str,
        speaker_id: str,
        language: str,
    ) -> str:
        model, tokenizer = self._load_vits(speaker_id, language)
        speaker_emb = torch.tensor(
            VoiceEnroller.load_embedding(speaker_id), dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        inputs = tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            kwargs = dict(inputs)
            if hasattr(model.config, "speaker_embeddings_dim"):
                kwargs["speaker_embeddings"] = speaker_emb
            output = model(**kwargs)

        waveform = output.waveform.squeeze().cpu().numpy()
        sr = model.config.sampling_rate
        return self._save_wav(waveform, sr)

    # ── YourTTS (zero-shot fallback) ──────────────────────────────────────────
    def _load_yourtts(self):
        if self._yourtts_pipe is None:
            from TTS.api import TTS
            self._yourtts_pipe = TTS(
                model_name=YOURTTS_MODEL,
                progress_bar=False,
                gpu=torch.cuda.is_available(),
            )
            print("[ClonedSynth] ✅ YourTTS loaded (zero-shot fallback)")
        return self._yourtts_pipe

    def _synthesize_yourtts(
        self,
        text: str,
        language: str,
        speaker_id: str,
        reference_audio: Optional[str] = None,
    ) -> str:
        from config.voice_clone_settings import CLONE_LANG_MAP
        tts = self._load_yourtts()
        lang_code = CLONE_LANG_MAP[language]["yourtts"]

        # Use any enrolled audio file as reference voice
        if reference_audio is None:
            profile = VoiceEnroller.load_profile(speaker_id)
            audio_files = profile.get("audio_files", [])
            if audio_files:
                reference_audio = audio_files[0]

        out_path = f"/tmp/{uuid.uuid4()}.wav"
        tts.tts_to_file(
            text=text,
            language=lang_code,
            speaker_wav=reference_audio,
            file_path=out_path,
        )
        return out_path

    # ── save helper ───────────────────────────────────────────────────────────
    @staticmethod
    def _save_wav(audio: np.ndarray, sr: int) -> str:
        if not isinstance(audio, np.ndarray):
            audio = np.array(audio, dtype=np.float32)
        audio = audio.astype(np.float32)
        out_path = f"/tmp/{uuid.uuid4()}.wav"
        sf.write(out_path, audio, sr, format="WAV", subtype="PCM_16")
        return out_path

    # ── public API ────────────────────────────────────────────────────────────
    def synthesize(
        self,
        text: str,
        language: str,
        speaker_id: str,
        reference_audio: Optional[str] = None,
        prefer_finetuned: bool = True,
    ) -> str:
        """
        Synthesize speech in a cloned voice.

        Parameters
        ----------
        text             : translated text to speak
        language         : target language ("english"|"hindi"|"french"|"spanish")
        speaker_id       : enrolled speaker identifier
        reference_audio  : optional path to raw audio (for YourTTS fallback)
        prefer_finetuned : if True, try fine-tuned VITS first; else use YourTTS

        Returns
        -------
        Path to output WAV file
        """
        if language not in CLONE_SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {language}")

        if not text or not text.strip():
            raise ValueError("Cannot synthesize empty text.")

        if prefer_finetuned and self._ckpt_path(speaker_id, language):
            print(f"[ClonedSynth] Using fine-tuned VITS ({speaker_id}/{language})")
            try:
                return self._synthesize_vits(text, speaker_id, language)
            except Exception as e:
                print(f"[ClonedSynth] ⚠️ VITS failed ({e}), falling back to YourTTS…")

        print(f"[ClonedSynth] Using YourTTS zero-shot ({speaker_id}/{language})")
        return self._synthesize_yourtts(text, language, speaker_id, reference_audio)
