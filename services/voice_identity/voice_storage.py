import uuid
from services.voice_identity.voice_encoder import VoiceEncoder
from services.database.qdrant_client import qdrant_client, COLLECTION_NAME

encoder = VoiceEncoder()

def store_voice(audio_path: str, speaker: str, language: str):
    embedding = encoder.encode(audio_path)

    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            {
                "id": str(uuid.uuid4()),
                "vector": embedding,
                "payload": {
                    "speaker": speaker,
                    "language": language,
                },
            }
        ],
    )
