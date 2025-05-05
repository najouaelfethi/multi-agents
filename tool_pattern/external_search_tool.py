import json
import wikipedia
from tool_pattern.tool import tool

@tool
def external_search(query: str, sentences: int = 3) -> str: #select 3 sentences then make summary
    """
    Performs a Wikipedia search and returns a summary for the query.

    Args:
        query (str): The search term.
        sentences (int): Number of sentences to return in the summary.

    Returns:
        str: A summary of the Wikipedia page content.
    """
    try:
        summary = wikipedia.summary(query, sentences=sentences)
        return json.dumps({"summary": summary})
    except wikipedia.exceptions.DisambiguationError as e:
        return json.dumps({"error": f"Disambiguation error. Options: {e.options}"})
    except wikipedia.exceptions.PageError:
        return json.dumps({"error": "No Wikipedia page found for your question."})
    except Exception as e:
        return json.dumps({"error": str(e)})
