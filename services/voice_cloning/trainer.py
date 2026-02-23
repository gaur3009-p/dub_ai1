import os
import json
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import List, Optional
from torch.utils.data import Dataset, DataLoader
from transformers import (
    VitsModel,
    VitsTokenizer,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW

from config.voice_clone_settings import (
    VITS_BASE_MODELS,
    CLONE_TRAIN_EPOCHS,
    CLONE_TRAIN_BATCH_SIZE,
    CLONE_TRAIN_LR,
    CLONE_CHECKPOINTS_DIR,
    CLONE_SUPPORTED_LANGUAGES,
)
from services.voice_cloning.enroller import VoiceEnroller


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class VoiceDataset(Dataset):
    """
    Simple dataset that loads (text, waveform) pairs for VITS fine-tuning.
    The transcripts are expected alongside audio:
        /path/to/audio.wav  →  /path/to/audio.txt
    If a .txt sidecar is missing, the utterance is skipped.
    """

    def __init__(
        self,
        audio_paths: List[str],
        tokenizer: VitsTokenizer,
        target_sr: int = 22_050,
        max_audio_len: int = 220_500,   # 10 s at 22 kHz
    ):
        self.samples = []
        self.tokenizer = tokenizer
        self.target_sr = target_sr
        self.max_audio_len = max_audio_len

        for ap in audio_paths:
            txt_path = Path(ap).with_suffix(".txt")
            if not txt_path.exists():
                print(f"  [Dataset] ⚠️  No transcript for {ap} – skipped.")
                continue
            transcript = txt_path.read_text().strip()
            if transcript:
                self.samples.append((ap, transcript))

        print(f"  [Dataset] {len(self.samples)} utterances loaded.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio_path, text = self.samples[idx]

        # ── waveform ─────────────────────────────────────────────────────────
        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.target_sr)
        waveform = waveform.squeeze(0)                  # (T,)
        if waveform.shape[-1] > self.max_audio_len:
            waveform = waveform[:self.max_audio_len]

        # ── text ─────────────────────────────────────────────────────────────
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "waveform": waveform,
        }


def _collate(batch):
    from torch.nn.utils.rnn import pad_sequence
    input_ids = pad_sequence([b["input_ids"] for b in batch], batch_first=True)
    attention_mask = pad_sequence([b["attention_mask"] for b in batch], batch_first=True)
    waveforms = pad_sequence([b["waveform"] for b in batch], batch_first=True)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "waveforms": waveforms,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class VoiceCloningTrainer:
    """
    Fine-tune VITS for each language using a speaker's enrolled audio.

    Usage
    -----
    trainer = VoiceCloningTrainer(speaker_id="aditya")
    trainer.train(
        language="hindi",
        audio_paths=["rec1.wav", "rec2.wav"],
        epochs=200,
    )
    """

    def __init__(self, speaker_id: str):
        self.speaker_id = speaker_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load speaker embedding (must enroll first)
        self.speaker_embedding = torch.tensor(
            VoiceEnroller.load_embedding(speaker_id), dtype=torch.float32
        ).unsqueeze(0).to(self.device)   # (1, 192)

        print(f"[Trainer] Speaker: '{speaker_id}' | Device: {self.device}")

    # ── helpers ───────────────────────────────────────────────────────────────
    def _ckpt_dir(self, language: str) -> Path:
        d = Path(CLONE_CHECKPOINTS_DIR) / self.speaker_id / language
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def _spectrogram_loss(pred_waveform: torch.Tensor, target_waveform: torch.Tensor) -> torch.Tensor:
        """L1 loss in mel-spectrogram space (80 mels, 1024 FFT)."""
        import torchaudio.transforms as T
        mel = T.MelSpectrogram(sample_rate=22_050, n_fft=1024, n_mels=80).to(pred_waveform.device)
        # Trim/pad to same length
        min_len = min(pred_waveform.shape[-1], target_waveform.shape[-1])
        pred_mel = mel(pred_waveform[..., :min_len])
        tgt_mel  = mel(target_waveform[..., :min_len])
        return torch.nn.functional.l1_loss(pred_mel, tgt_mel)

    # ── core training ─────────────────────────────────────────────────────────
    def train(
        self,
        language: str,
        audio_paths: List[str],
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        lr: Optional[float] = None,
    ) -> str:
        """
        Fine-tune for one language.

        Returns path to the saved checkpoint directory.
        """
        if language not in CLONE_SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Language '{language}' not supported. "
                f"Choose from {CLONE_SUPPORTED_LANGUAGES}"
            )

        epochs     = epochs     or CLONE_TRAIN_EPOCHS
        batch_size = batch_size or CLONE_TRAIN_BATCH_SIZE
        lr         = lr         or CLONE_TRAIN_LR

        base_model_name = VITS_BASE_MODELS[language]
        print(f"\n[Trainer] 🚀 Fine-tuning for '{language}' using {base_model_name}")

        # ── Load base model + tokenizer ───────────────────────────────────────
        tokenizer = VitsTokenizer.from_pretrained(base_model_name)
        model = VitsModel.from_pretrained(base_model_name).to(self.device)

        # Inject speaker embedding dimension via model config (if supported)
        # VITS HF models expose speaker_embeddings_dim
        if hasattr(model.config, "speaker_embeddings_dim"):
            print(f"  [Trainer] Speaker embedding dim: {model.config.speaker_embeddings_dim}")

        # ── Dataset ───────────────────────────────────────────────────────────
        dataset = VoiceDataset(audio_paths, tokenizer)
        if len(dataset) == 0:
            raise RuntimeError(
                "No valid (audio, transcript) pairs found. "
                "Place .txt sidecar files alongside each WAV."
            )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=_collate,
            num_workers=0,
        )

        # ── Optimizer + Scheduler ─────────────────────────────────────────────
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        total_steps = epochs * len(loader)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=max(1, total_steps // 10),
            num_training_steps=total_steps,
        )

        # ── Training loop ─────────────────────────────────────────────────────
        model.train()
        best_loss = float("inf")
        ckpt_dir = self._ckpt_dir(language)

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            for step, batch in enumerate(loader):
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                target_wav     = batch["waveforms"].to(self.device)

                # Forward pass – generate waveform
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    speaker_embeddings=self.speaker_embedding.expand(
                        input_ids.shape[0], -1
                    ) if hasattr(model.config, "speaker_embeddings_dim") else None,
                )
                pred_wav = outputs.waveform  # (B, T_pred)

                # Loss: mel-spectrogram reconstruction
                loss = self._spectrogram_loss(pred_wav, target_wav)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / max(len(loader), 1)

            if epoch % 10 == 0 or epoch == 1:
                print(f"  Epoch {epoch:4d}/{epochs}  loss={avg_loss:.4f}")

            # ── Save best checkpoint ──────────────────────────────────────────
            if avg_loss < best_loss:
                best_loss = avg_loss
                model.save_pretrained(str(ckpt_dir))
                tokenizer.save_pretrained(str(ckpt_dir))

        # ── Save training metadata ─────────────────────────────────────────────
        meta = {
            "speaker_id": self.speaker_id,
            "language": language,
            "base_model": base_model_name,
            "epochs": epochs,
            "best_loss": round(best_loss, 6),
            "checkpoint_dir": str(ckpt_dir),
            "phase": "trained",
        }
        with open(ckpt_dir / "train_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"\n[Trainer] ✅ '{language}' checkpoint → {ckpt_dir}  (loss={best_loss:.4f})")
        return str(ckpt_dir)

    # ── train all languages ───────────────────────────────────────────────────
    def train_all_languages(
        self,
        audio_paths_by_lang: dict,
        epochs: Optional[int] = None,
    ) -> dict:
        """
        Train for multiple languages in one call.

        Parameters
        ----------
        audio_paths_by_lang : {
            "english": ["en1.wav", "en2.wav"],
            "hindi":   ["hi1.wav"],
            "french":  ["fr1.wav", "fr2.wav"],
            "spanish": ["es1.wav"],
        }
        """
        results = {}
        for lang, paths in audio_paths_by_lang.items():
            try:
                ckpt = self.train(language=lang, audio_paths=paths, epochs=epochs)
                results[lang] = {"status": "ok", "checkpoint": ckpt}
            except Exception as e:
                results[lang] = {"status": "error", "error": str(e)}
                print(f"[Trainer] ❌ Training failed for '{lang}': {e}")
        return results
