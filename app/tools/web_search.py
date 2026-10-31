
from langchain_tavily import TavilySearch


web_search = TavilySearch(
    max_results=5,
    topic="general",
)


def search_web(query: str):
    """
    Search the web and return relevant results.
    """

    print(f"\n🌐 Searching web: {query}")

    results = web_search.invoke({
        "query": query
    })

    return results

