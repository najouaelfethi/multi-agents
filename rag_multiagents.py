from config.settings import Config
from multiagent_pattern.agent import Agent
from multiagent_pattern.crewai import Crew
#from tool_pattern.chunking_tool import chunk_document
#from tool_pattern.embedding_storage_tool import embed_chunks 
from core.chunker import DocumentChunker
from core.document_loader import DocumentLoader
from processors.image_processor import ImageTextProcessor
from processors.text_processor import TextProcessor
from storage.pinecone_vector import VectorStore
from tool_pattern.internal_search_tool import internal_search
from tool_pattern.external_search_tool import external_search
from storage.pinecone_vector import VectorStore
import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings


load_dotenv()

model_embedding = OpenAIEmbeddings(openai_api_key=Config.openai_key)

chunker = DocumentChunker(model_embedding)
text_processor = TextProcessor()
image_processor = ImageTextProcessor()  # Si jamais tu veux embed aussi des images
vector_store = VectorStore()

# --------------- User Input ---------------

print("\n Welcome to the AI Assistant")
user_query = input("What would you like to ask? ")

choice = input("Do you want to search using internal documents or external sources? (Type 'internal' or 'external'): ").strip().lower()

uploaded_doc_path = None
if choice == "internal":
    uploaded_doc_path = input("Please enter the file path of the document to process: ").strip()
    
    #Automatic document processing, no tools needed here
    if uploaded_doc_path and os.path.exists(uploaded_doc_path):
        print("\nProcessing document...")
        try:
            # Detect document type
            doc_type = DocumentLoader.detect_type(uploaded_doc_path)
            
            # Chunk the document
            chunks = chunker.chunk_document(file_path=uploaded_doc_path,doc_type=doc_type)
            
            # Create namespace based on document name
            namespace = os.path.splitext(os.path.basename(uploaded_doc_path))[0]

            #Text chunks embedding
            text_vectors = []
            for i, chunk in enumerate(chunks):
                if chunk.page_content.strip(): 
                    processed = text_processor.process_text(chunk.page_content)
                    text_vectors.append({
                        "id": f"text_{i}",
                        "values": processed["embedding"],
                        "metadata": {
                            "type": "text",
                            "content": processed["content"][:100],
                            "source": uploaded_doc_path
                        }
                    })

            #Image chunks embedding
            output_dir = "extracted_images"
            image_paths = image_processor.extract_images_from_document(uploaded_doc_path, output_dir)

            image_vectors = []
            extracted = image_processor.extract_text_from_images(image_paths)
            for j, item in enumerate(extracted):
                image_vectors.append({
                    "id": f"image_{j}",
                    "values": item["embedding"],
                    "metadata": {
                        "type": "image",
                        "content": item["content"][:100],
                        "source": item["image_path"]
                    }
                })


            #Store vectors in Pinecone
            all_vectors = text_vectors + image_vectors
            if all_vectors:
                vector_store.store_vectors(all_vectors, namespace=namespace)
                print(f"Successfully stored {len(all_vectors)} chunks in namespace '{namespace}'")
            else:
                print("No valid chunks were processed.")
                
        except Exception as e:
            print(f"Error processing document: {str(e)}")
    else:
        print("Error: Document path is invalid or file does not exist.")

# --------------- CREW & AGENTS ---------------

with Crew() as crew:
    # Router Agent: Analyse user input and make decision wich pipeline to use: Internal or External
    router_agent = Agent(
        name="RouterAgent",
        backstory="You decide whether the query should be answered using internal company documents or external web sources.",
        task_description=f"The client chose '{choice}'. Use this to decide whether to use internal or external search.",
        task_expected_output="Return either 'internal' or 'external'.",
        tools=[]
    )

    # Retriever Agent
    retriever_agent = Agent(
        name="RetrieverAgent",
        backstory="You retrieve relevant information either from internal documents or external sources.",
        task_description="If internal search was selected, search the internal vector database. Otherwise, perform an external search.",
        task_expected_output="Provide retrieved information.",
        tools=[internal_search, external_search],
    )

    # Answer Generator Agent
    answer_agent = Agent(
        name="AnswerAgent",
        backstory="You generate a final answer based on the retrieved information.",
        task_description=f"Use the retrieved information to answer this question: '{user_query}', if you don't have any information, say 'No information found'.",
        task_expected_output="Provide a good answer.",
        tools=[]
    )

    # -------------- DEPENDENCIES ---------------

    #router_agent >> doc_processor_agent  #if internal is chosen
    router_agent >> retriever_agent
    retriever_agent >> answer_agent

    # --------------- Pass User Context ---------------

    router_agent.receive_context(f"User query: {user_query}\nClient choice: {choice}")

    if choice == "internal" and uploaded_doc_path:
        #doc_processor_agent.receive_context(f"Document path: {uploaded_doc_path}")
        retriever_agent.receive_context({"namespace": namespace, "search_type": "internal"})
    else:
        retriever_agent.receive_context("Use external search.")

    # -------------RUN CREW ---------------

    crew.run()

#documents/attention_pdf.pdf