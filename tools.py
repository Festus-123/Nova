from langchain_core.tools import tool
from ddgs import DDGS
from pathlib import Path
import wikipedia

@tool
def wikipedia_search(query: str) -> str:
    """
    Search Wikipedia for information about a topic.
    """

    try:
        return wikipedia.summary(query, sentences=5)

    except wikipedia.DisambiguationError as e:
        return f"Multiple matches found: {e.options[:5]}"

    except Exception as e:
        return str(e)


@tool
def web_search(query: str) -> str:
    """
    Search the web for recent information.
    """

    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=5))

    if not results:
        return "No results found."

    output = []

    for result in results:
        output.append(
            f"""
Title: {result['title']}
URL: {result['href']}
Body: {result['body']}
"""
        )

    return "\n".join(output)


@tool
def save_report(filename: str, content: str) -> str:
    """
    Save research results to a text file.
    """

    Path(filename).write_text(content, encoding="utf-8")

    return f"Saved report to {filename}"


@tool
def read_report(filename: str) -> str:
    """
    Read a saved report.
    """

    return Path(filename).read_text(encoding="utf-8")

