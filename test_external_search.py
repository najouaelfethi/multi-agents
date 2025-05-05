from tool_pattern.external_search_tool import external_search


query="what is happiness?"
args = {"query":query, "sentences":3}
response = external_search.run(**args)
print(response)
