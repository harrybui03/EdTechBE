from typing import List, Tuple, TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict, total=False):
    """
    Represents a state of a graph.

    Attributes:
        question: Question
        generation: LLM Generation
        use_web_search: wether to use web search
        documents: List of documents
        sources: List of metadata for surfaced documents
        user_id: Optional user identifier for permission checks
        lesson_id: Optional lesson ID to filter documents to specific lesson
        chat_history: List of tuples (question, answer) for conversation context
        regeneration_count: Number of times generation has been regenerated (to prevent infinite loops)
    """

    question: str
    generation: str
    use_web_search: bool
    documents: List[Document]
    sources: List[dict]
    user_id: str
    lesson_id: str
    is_platform_question: bool  # Flag to indicate if this is a platform/knowledge-base question
    chat_history: List[Tuple[str, str]]
    regeneration_count: int
    web_search_count: int  # Track number of web search attempts to prevent loops
    original_language: str  # Language of the original question (e.g., 'vi', 'en')
