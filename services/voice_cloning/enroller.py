import os
import uuid
import json
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import List, Optional

from speechbrain.pretrained import EncoderClassifier
from config.voice_clone_settings import (
    SPEAKER_ENCODER_MODEL,
    CLONE_MIN_ENROLL_SECONDS,
    CLONE_EMBEDDING_DIM,
    VOICE_PROFILES_DIR,
)

# ── Lazy-load Qdrant (optional) ───────────────────────────────────────────────
try:
    from services.database.qdrant_client import qdrant_client, COLLECTION_NAME
    _QDRANT_OK = True
except Exception:
    _QDRANT_OK = False


class VoiceEnroller:
    """Enroll a speaker voice into the system (Phase 0)."""

    _encoder: Optional[EncoderClassifier] = None   # singleton

    # ── encoder ───────────────────────────────────────────────────────────────
    @classmethod
    def _get_encoder(cls) -> EncoderClassifier:
        if cls._encoder is None:
            cls._encoder = EncoderClassifier.from_hparams(
                source=SPEAKER_ENCODER_MODEL,
                run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
                savedir="pretrained_models/spk_enc",
            )
        return cls._encoder

    # ── audio helpers ─────────────────────────────────────────────────────────
    @staticmethod
    def _load_audio(path: str, target_sr: int = 16_000) -> torch.Tensor:
        waveform, sr = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        return waveform  # (1, T)

    @staticmethod
    def _total_seconds(audio_paths: List[str]) -> float:
        total = 0.0
        for p in audio_paths:
            info = torchaudio.info(p)
            total += info.num_frames / info.sample_rate
        return total

    # ── embedding ─────────────────────────────────────────────────────────────
    def _embed(self, audio_paths: List[str]) -> np.ndarray:
        """Compute mean embedding over all provided audio files."""
        encoder = self._get_encoder()
        embeddings = []
        for path in audio_paths:
            waveform = self._load_audio(path)               # (1, T)
            with torch.no_grad():
                emb = encoder.encode_batch(waveform)        # (1, 1, D)
            embeddings.append(emb.squeeze().cpu().numpy())  # (D,)
        mean_emb = np.mean(embeddings, axis=0)
        # L2-normalise
        mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)
        return mean_emb  # (192,)

    # ── public API ────────────────────────────────────────────────────────────
    def enroll(
        self,
        speaker_id: str,
        audio_paths: List[str],
        languages: List[str],
        profile_meta: Optional[dict] = None,
    ) -> dict:
        """
        Enroll a speaker.

        Parameters
        ----------
        speaker_id   : unique identifier for the speaker (e.g. "aditya")
        audio_paths  : list of WAV/MP3 paths from this speaker
        languages    : languages the voice will be used to dub into
                       e.g. ["english", "hindi", "french", "spanish"]
        profile_meta : optional extra metadata (name, age, …)

        Returns
        -------
        dict with keys: speaker_id, embedding_path, profile_path, enrolled_ok
        """
        if not audio_paths:
            raise ValueError("Provide at least one audio file for enrollment.")

        duration = self._total_seconds(audio_paths)
        if duration < CLONE_MIN_ENROLL_SECONDS:
            raise ValueError(
                f"Need at least {CLONE_MIN_ENROLL_SECONDS}s of speech; "
                f"got {duration:.1f}s."
            )

        print(f"[Enroller] Computing speaker embedding for '{speaker_id}' "
              f"({duration:.1f}s of audio, {len(audio_paths)} file(s))…")

        embedding: np.ndarray = self._embed(audio_paths)   # (192,)

        # ── persist embedding ─────────────────────────────────────────────────
        profile_dir = Path(VOICE_PROFILES_DIR) / speaker_id
        profile_dir.mkdir(parents=True, exist_ok=True)

        emb_path = profile_dir / "embedding.npy"
        np.save(str(emb_path), embedding)

        profile = {
            "speaker_id": speaker_id,
            "languages": languages,
            "embedding_dim": int(embedding.shape[0]),
            "audio_files": audio_paths,
            "duration_seconds": round(duration, 2),
            "phase": "enrolled",
            **(profile_meta or {}),
        }
        profile_path = profile_dir / "profile.json"
        with open(profile_path, "w") as f:
            json.dump(profile, f, indent=2)

        print(f"[Enroller] Embedding saved → {emb_path}")

        # ── upsert to Qdrant ──────────────────────────────────────────────────
        if _QDRANT_OK:
            try:
                qdrant_client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=[{
                        "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, speaker_id)),
                        "vector": embedding.tolist(),
                        "payload": {
                            "speaker_id": speaker_id,
                            "languages": languages,
                            "phase": "enrolled",
                        },
                    }],
                )
                print("[Enroller] ✅ Qdrant upsert complete.")
            except Exception as e:
                print(f"[Enroller] ⚠️ Qdrant upsert failed: {e}")
        else:
            print("[Enroller] ⚠️ Qdrant unavailable – embedding stored locally only.")

        return {
            "speaker_id": speaker_id,
            "embedding_path": str(emb_path),
            "profile_path": str(profile_path),
            "duration_seconds": round(duration, 2),
            "enrolled_ok": True,
        }

    @staticmethod
    def load_embedding(speaker_id: str) -> np.ndarray:
        """Load a previously saved embedding from disk."""
        path = Path(VOICE_PROFILES_DIR) / speaker_id / "embedding.npy"
        if not path.exists():
            raise FileNotFoundError(
                f"No embedding found for speaker '{speaker_id}'. "
                "Run enroll() first."
            )
        return np.load(str(path))

    @staticmethod
    def load_profile(speaker_id: str) -> dict:
        """Load a previously saved profile JSON from disk."""
        path = Path(VOICE_PROFILES_DIR) / speaker_id / "profile.json"
        if not path.exists():
            raise FileNotFoundError(f"No profile for speaker '{speaker_id}'.")
        with open(path) as f:
            return json.load(f)
