import librosa
import numpy as np

class VoiceEncoder:
    def encode(self, audio_path: str):
        audio, sr = librosa.load(audio_path, sr=16000)
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=40
        )
        embedding = mfcc.mean(axis=1)
        return embedding.tolist()
