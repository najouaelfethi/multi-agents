import json
from tool_pattern.tool import tool
from storage.pinecone_vector import VectorStore  # Adjust the import based on your folder name

vs = VectorStore()

@tool
def internal_search(query: str, namespace: str = "default", top_k: int = 3) -> str:
    """
    Perform a semantic search over internal documents using Pinecone.
    
    Args:
        query (str): The search query.
        namespace (str): The namespace to search in.
        top_k (int): Number of top results to return.
    
    Returns:
        str: A JSON string with top search results and metadata.
    """
    try:
        # Embed the query
        query_vector = vs.embed_query(query)
        # Search Pinecone
        results = vs.search_vectors(query_vector, namespace, top_k)
        return json.dumps(results.to_dict())
    except Exception as e:
        return json.dumps({"error": str(e)})
