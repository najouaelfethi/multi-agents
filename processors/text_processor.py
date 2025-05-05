from typing import List
from langchain_openai import OpenAIEmbeddings
from config.settings import Config

class TextProcessor:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=Config.openai_key
        )
    
    def process_text(self, text: str) -> dict:
        embedding = self.embeddings.embed_query(text)
        
        return {
            "type": "text",
            "content": text,
            "embedding": embedding
        }