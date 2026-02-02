from qdrant_client import QdrantClient
from config.settings import QDRANT

qdrant_client = QdrantClient(
    url=QDRANT["url"],
    api_key=QDRANT["api_key"],
)

COLLECTION_NAME = QDRANT["collection"]

def get_collections():
    return qdrant_client.get_collections()
