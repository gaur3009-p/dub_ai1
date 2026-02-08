import uuid
import torch
import librosa
from services.database.qdrant_client import qdrant_client, COLLECTION_NAME
from transformers import AutoModel

model = AutoModel.from_pretrained("coqui/XTTS-v2")

def store_voice(audio_path: str, speaker: str, language: str):
    audio, sr = librosa.load(audio_path, sr=16000)
    audio = torch.tensor(audio).unsqueeze(0)

    with torch.no_grad():
        embedding = model.get_speaker_embedding(audio)

    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=[{
            "id": str(uuid.uuid4()),
            "vector": embedding.squeeze().tolist(),
            "payload": {
                "speaker": speaker,
                "language": language,
            }
        }]
    )

    return embedding
