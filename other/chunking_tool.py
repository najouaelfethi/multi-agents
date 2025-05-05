from core.chunker import DocumentChunker
from tool_pattern.tool import tool
from config.settings import Config

@tool
def chunk_document(file_path: str, doc_type: str):
    """
    Chunks a document into smaller pieces based on its type.
    
    Args:
        file_path (str): Path to the document
        doc_type (str): Type of document (e.g., '.pdf', '.docx', '.txt', '.pptx')
    
    Returns:
        List[Document]: List of document chunks
    """
    chunker = DocumentChunker(Config.model_embeddings)
    return chunker.chunk_document(file_path, doc_type)