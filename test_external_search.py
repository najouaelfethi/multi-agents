from tool_pattern.external_search_tool import external_search


query="what is happiness?"
response = external_search.run(query, sentences=3)
print(response)
