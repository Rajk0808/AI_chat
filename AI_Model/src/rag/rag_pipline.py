import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
from retriever import retrieve_context

class RAGPipeline:
    def retriever(self,
                  query : str,
                  top_k : int
                  ):
        return retrieve_context(query=query, top_k=top_k)


def should_use_rag(query) -> bool:
    NON_RAG_PREFIXES = (
        "what is",
        "who is",
        "define",
        "meaning of",
        "introduction to",
        "explain"
    )

    MEDICAL_KEYWORDS = (
        "symptom",
        "disease",
        "treatment",
        "fever",
        "vomiting",
        "diarrhea",
        "skin",
        "infection",
        "breathing",
        "poison"
    )

    if query.startswith(NON_RAG_PREFIXES):
        return False

    if any(word in query for word in MEDICAL_KEYWORDS):
        return True

    return False
