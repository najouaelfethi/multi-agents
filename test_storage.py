from processors.text_processor import TextProcessor
from processors.image_processor import ImageTextProcessor
from core.document_loader import DocumentLoader
from core.chunker import DocumentChunker
from storage.pinecone_vector import VectorStore
from config.settings import Config

if __name__ == "__main__":
    file_path = "./documents/attention_pdf.pdf"
    #file_path="./documents/Presentation IDEO Studio Manager.pptx"
    namespace = "pdf" #testing with pdf
    #namespace="pdf-multimodal"
    #detect document
    doc_type = DocumentLoader.detect_type(file_path)

    print("Processing text...")
    #Load document
    docs = DocumentLoader.load_document(file_path)

    #Modality Text: Process text+Embedding
    text_processor = TextProcessor()
    chunker = DocumentChunker(embeddings=text_processor.embeddings)
    chunks = chunker.chunk_document(file_path, doc_type=doc_type)

    text_vectors = []
    for i, chunk in enumerate(chunks):
        text = chunk.page_content if hasattr(chunk, "page_content") else chunk
        processed = text_processor.process_text(text)
        text_vectors.append({
            "id": f"text_{i}",
            "values": processed["embedding"],
            "metadata": {
                "type": "text",
                "content": processed["content"][:100],
                "source": file_path
            }
        })

    #Modality Image: Process Images: Extract+Embedding
    print("Processing images...")
    img_processor = ImageTextProcessor()
    image_paths = img_processor.extract_images_from_document(file_path, output_dir="./extracted_images")
    image_results = img_processor.extract_text_from_images(image_paths)

    image_vectors = []
    for i, r in enumerate(image_results):
        image_vectors.append({
            "id": f"image_{i}",
            "values": r["embedding"],
            "metadata": {
                "type": "image",
                "image_path": r["image_path"],
                "content": r["content"][:100],
                "source": file_path
            }
        })

    #store vectors in Pinecone
    print("Storing vectors in Pinecone...")
    store = VectorStore()
    all_vectors = text_vectors + image_vectors  
    store.store_vectors(all_vectors, namespace=namespace)
    print(f"Stored {len(text_vectors)} text chunks.")
    print(f"Stored {len(image_vectors)} image chunks.")
    print(f"Total stored in namespace '{namespace}': {len(all_vectors)} vectors.")


