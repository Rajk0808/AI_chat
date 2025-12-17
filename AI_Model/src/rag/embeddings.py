from typing import List
from openai import OpenAI
import config.constants as constants
class EmbeddingServices:
    """
    Handles embedding genration for query
    """

    def __init__(self):
        self.client=OpenAI(api_key=constants.OPENAIAPI)
        self.model = constants.EMBEDDING_MODEL

    def embed_query(self,query:str)->List[float]:

        if not query or query.strip():
            raise ValueError("QUestion text is empty")

        response=self.client.embeddings.create(
            model=self.model,
            input=query
        )
        return response.data[0].embedding

_embeding_service=EmbeddingServices()
def embed_query(query:str)->List[float]:
    return _embeding_service.embed_query(query)
    

