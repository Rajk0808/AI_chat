from embeddings import embed_query
from pinecone import Pinecone
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config.constants as constants
#api = pcsk_21TWTB_4g6huMtcu3YtLfa1ZVXKcdrum5H6GPXrGiwg1UNg5jvskmgM2MmAgr1VwVHXfpw
class PineconeRetriever:
    def __init__(self):
        self.pc = Pinecone(api_key=constants.PINECONE_API_KEY)
        self.index = self.pc.Index(name=constants.PINECONE_INDEX_NAME)

    def retrive_context(self, query: str, top_k: int):
        query_vector = embed_query(query)
      
        result = self.index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True
        )     

        context = []

        for match in result["matches"]:
            metadata = match.get("metadata", {})
            context.append({
                "text": (
                    metadata.get("text")
                    or metadata.get("chunk_text")
                    or metadata.get("content")
                    or ""
                ),
                "score": match.get("score"),
                "source": metadata.get("source", "unknown")
            })

        return context