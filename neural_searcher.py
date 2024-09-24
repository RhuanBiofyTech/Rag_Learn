
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


class NeuralSearcher:
    def __init__(self, collection_name):
        self.collection_name = collection_name
        # Initialize encoder model
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        # initialize Qdrant client
        self.qdrant_client = QdrantClient(url="http://localhost:6333", api_key="testando_321")
    
    def search(self, text: str):
        vector = self.model.encode(text).tolist()
        
        search_result = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=vector,
            query_filter=None,
            limit=5,
        ).points

        payloads = [hit.payload for hit in search_result]
        return payloads