from multiagent_pattern.agent import Agent
from multiagent_pattern.crewai import Crew
from tool_pattern.chunking_tool import chunk_document
from tool_pattern.embedding_tool import embed_chunks 
from tool_pattern.internal_search_tool import internal_search
from tool_pattern.external_search_tool import external_search

# --------------- User Input ---------------

print("\n Welcome to the AI Assistant")
user_query = input("What would you like to ask? ")

choice = input("Do you want to search using internal documents or external sources? (Type 'internal' or 'external'): ").strip().lower()

uploaded_doc_path = None
if choice == "internal":
    uploaded_doc_path = input("Please enter the file path of the document to process: ").strip()

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

    # Document Processor Agent: Chunking and Embedding for Internal documents
    doc_processor_agent = Agent(
        name="DocumentProcessorAgent",
        backstory="You process client documents by chunking and embedding them for search.",
        task_description="Chunk and embed the provided document for future search.",
        task_expected_output="Chunks stored and embeddings created.",
        tools=[chunk_document, embed_chunks],
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
        task_description=f"Use the retrieved information to answer this question: '{user_query}'",
        task_expected_output="Provide a good answer.",
        tools=[]
    )

    # -------------- DEPENDENCIES ---------------

    router_agent >> doc_processor_agent  #if internal is chosen
    router_agent >> retriever_agent
    retriever_agent >> answer_agent

    # --------------- Pass User Context ---------------

    router_agent.receive_context(f"User query: {user_query}\nClient choice: {choice}")

    if choice == "internal" and uploaded_doc_path:
        doc_processor_agent.receive_context(f"Document path: {uploaded_doc_path}")
        retriever_agent.receive_context("Use internal search.")
    else:
        retriever_agent.receive_context("Use external search.")

    # -------------RUN CREW ---------------

    crew.run()

