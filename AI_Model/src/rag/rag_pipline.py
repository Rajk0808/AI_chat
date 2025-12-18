import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
from retriever import PineconeRetriever as retriever

class RAGPipeline:
    def retriever(self,
                  query : str,
                  top_k : int
                  ):
        return retriever.retrive_context(self = retriever(),query=query, top_k = top_k)


