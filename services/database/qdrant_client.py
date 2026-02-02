from qdrant_client import QdrantClient
from config.settings import QDRANT

qdrant = QdrantClient(
    host=QDRANT["host"],
    port=QDRANT["port"]
)

COLLECTION = QDRANT["collection"]

def init_collection(vector_size=256):
    collections = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION not in collections:
        qdrant.recreate_collection(
            collection_name=COLLECTION,
            vectors_config={"size": vector_size, "distance": "Cosine"}
        )
