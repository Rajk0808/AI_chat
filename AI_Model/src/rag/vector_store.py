from langchain_community.vectorstores import Chroma

def create_vector_store(chunk,emb):
    db = Chroma.from_documents(
        documents = chunk,
        embedding = emb,
        collection_name="pet_db",
        persist_directory="./chroma_pet"
    )
    
    db.persist()
    print('Documents Stored')