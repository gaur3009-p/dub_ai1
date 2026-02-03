from qdrant_client import QdrantClient
from config.settings import QDRANT

qdrant_client = QdrantClient(
    url=QDRANT["url"],
    api_key=QDRANT["api_key"],
)

COLLECTION_NAME = QDRANT["collection"]
