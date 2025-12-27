import logging
from typing import Any, Dict

from langchain_core.documents import Document
from langchain_community.tools.tavily_search import TavilySearchResults

from graph.state import GraphState
from graph.nodes.web_search_validator import validate_web_search
from graph.nodes.retrieve import _detect_language, _translate_to_english

logger = logging.getLogger("agentic_rag.web_search")


web_search_tool = TavilySearchResults(max_results=3)


def web_search(state: GraphState) -> Dict[str, Any]:
    """
    Search the web for documents.
    Only searches if question is validated as appropriate for web search.

    Args:
        state (dict): The current state of the graph.

    Returns:
        state (dict): A dictionary containing the retrieved documents and the question
    """
    print("---WEB SEARCH---")
    logger.info("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"] or []  # only relevant documents
    web_search_count = (state.get("web_search_count") or 0) + 1  # Increment web search counter

    # Validate if web search is appropriate for this question
    is_valid, reason = validate_web_search(question)
    
    if not is_valid:
        msg1 = f"---WEB SEARCH REJECTED: {reason}---"
        msg2 = "---RETURNING WITHOUT WEB SEARCH---"
        print(msg1)
        print(msg2)
        logger.info(msg1)
        logger.info(msg2)
        # Return without web search - let existing documents handle the question
        return {
            "documents": documents,
            "question": question,
            "user_id": state.get("user_id"),
            "lesson_id": state.get("lesson_id"),  # Preserve lesson_id
            "chat_history": state.get("chat_history", []),
            "regeneration_count": 0,  # Reset when starting web search flow
            "web_search_count": web_search_count,
            "original_language": state.get("original_language"),  # Preserve original language
        }

    print(f"---WEB SEARCH VALIDATED: {reason}---")
    
    # Translate question to English for better web search results
    original_language = state.get("original_language") or _detect_language(question)
    search_query = question  # Default to original question
    
    if original_language != 'en':
        print(f"---TRANSLATING WEB SEARCH QUERY TO ENGLISH (original: {original_language})---")
        search_query = _translate_to_english(question)
    else:
        print("---WEB SEARCH QUERY IS IN ENGLISH, NO TRANSLATION NEEDED---")
    
    # Enhance query to focus on educational/technical content
    # Add context to search query to get more relevant results
    enhanced_query = f"{search_query} educational course tutorial"
    
    try:
        tavily_results = web_search_tool.invoke({"query": enhanced_query})
    except Exception as e:
        print(f"---WEB SEARCH ERROR: {e}---")
        # Return without web search on error
        return {
            "documents": documents,
            "question": question,
            "user_id": state.get("user_id"),
            "lesson_id": state.get("lesson_id"),  # Preserve lesson_id
            "chat_history": state.get("chat_history", []),
            "regeneration_count": 0,  # Reset when starting web search flow
            "web_search_count": web_search_count,
            "original_language": state.get("original_language"),  # Preserve original language
        }

    # get one huge string with all the results
    tavily_results_joined = "\n".join([res["content"] for res in tavily_results])

    # create a document object
    web_search_result = Document(
        page_content=tavily_results_joined,
        metadata={
            "doc_type": "web_search",
            "source": "tavily",
            "requires_enrollment": False,
            "search_query": question,
        },
    )

    # append web search to the list of documents
    documents.append(web_search_result)

    return {
        "documents": documents,
        "question": question,
        "user_id": state.get("user_id"),
        "lesson_id": state.get("lesson_id"),  # Preserve lesson_id
        "chat_history": state.get("chat_history", []),
        "regeneration_count": 0,  # Reset when starting web search flow
        "web_search_count": web_search_count,
        "original_language": state.get("original_language"),  # Preserve original language
    }
