from typing import Union, List
from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader
)
from config.settings import Config #contient supported doc types
from langchain.docstore.document import Document
from pptx import Presentation  


class SimpleTextLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Document]:
        with open(self.file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return [Document(page_content=text)]

class DocumentLoader:
    @staticmethod #using this method without instancier class
    def detect_type(file_path: str) -> str:
        return Path(file_path).suffix.lower()#return extension of file

    @staticmethod
    def load_document(file_path: str) -> List[dict]:
        doc_type = DocumentLoader.detect_type(file_path)#detect type
        
        if doc_type not in Config.SUPPORTED_DOC_TYPES:
            raise ValueError(f"Unsupported document type: {doc_type}")
        
        if doc_type == ".pdf":
            loader = PyPDFLoader(file_path)
            return loader.load()

        elif doc_type == ".docx":
            loader = UnstructuredWordDocumentLoader(file_path)
            return loader.load()

        elif doc_type == ".pptx":
            loader = Presentation(file_path)
            return loader

           
        elif doc_type == ".txt":
            loader = SimpleTextLoader(file_path)
            return loader.load()

