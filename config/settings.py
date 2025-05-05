import os
import openai
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings

load_dotenv()

class Config:

    #API Keys
    openai_key = os.getenv("OPENAI_API_KEY")
    openai_org = os.getenv("OPENAI_ORG_ID")
    pinecone_key = os.getenv("PINECONE_API_KEY")
    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

    #Storage Settings
    pc = Pinecone(api_key=pinecone_key)
    index = pc.Index(pinecone_index_name)

    model_embeddings = OpenAIEmbeddings(openai_api_key=openai_key,openai_organization=openai_org)

    #Model Settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    SUPPORTED_DOC_TYPES = [".pdf", ".docx", ".pptx", ".txt", ".xlsx", ".csv", ".json"]
    SUPPORTED_IMAGE_TYPES = [".jpg", ".jpeg", ".png"]