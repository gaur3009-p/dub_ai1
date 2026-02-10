import uuid
from services.database.qdrant_client import qdrant_client, COLLECTION_NAME


def store_voice(audio_path: str, speaker: str, language: str):
    """
    Store speaker reference audio for Bark cloning
    """

    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            {
                "id": str(uuid.uuid4()),
                "vector": [0.0],  # dummy vector (Bark does not need embeddings)
                "payload": {
                    "speaker": speaker,
                    "language": language,
                    "audio_path": audio_path,
                },
            }
        ],
    )
