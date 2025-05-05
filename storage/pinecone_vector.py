from pinecone import Pinecone
from config.settings import Config
from typing import List
from langchain_openai import OpenAIEmbeddings


class VectorStore:
    def __init__(self):
        self.pc = Pinecone(api_key=Config.pinecone_key)
        self.index = self.pc.Index(Config.pinecone_index_name)
        self.embedder = OpenAIEmbeddings(openai_api_key=Config.openai_key)

    
    def store_vectors(self, vectors: List[dict], namespace: str):
        #Store vectors in Pinecone
        self.index.upsert(vectors=vectors, namespace=namespace)
    
    def search_vectors(self, query_vector: List[float], namespace: str, top_k: int = 3):
        #Search vectors in Pinecone
        return self.index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace
        )
    
    def embed_query(self, query: str) -> List[float]:
        return self.embedder.embed_query(query)