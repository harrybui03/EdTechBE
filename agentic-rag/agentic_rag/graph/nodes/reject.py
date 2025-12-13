"""
Handle rejection of questions unrelated to courses/education.
Provides polite response without unnecessary API calls.
"""

from typing import Any, Dict

from graph.state import GraphState


def reject_unrelated_question(state: GraphState) -> Dict[str, Any]:
    """
    Handle questions that are unrelated to courses/education.
    Provides a polite response without going through the full RAG pipeline.
    
    Args:
        state: Current state of the graph
        
    Returns:
        Dictionary containing rejection message and empty sources
    """
    print("---REJECT UNRELATED QUESTION---")
    question = state["question"]
    chat_history = state.get("chat_history", [])
    
    # Polite rejection message
    rejection_message = """I'm an AI assistant focused on helping with course-related questions and educational content on this e-learning platform.

I can help you with:
- Questions about course content and lessons
- How to use the platform (creating courses, enrolling, etc.)
- Technical topics covered in courses
- Learning and educational topics

For questions about weather, news, sports, or other general topics, I'm not able to help. Please ask me about courses, learning, or platform usage instead.

How can I help you with your learning today?"""
    
    # Update chat history
    updated_history = list(chat_history) if chat_history else []
    updated_history.append((question, rejection_message))
    
    return {
        "generation": rejection_message,
        "documents": [],  # No documents for rejected questions
        "question": question,
        "sources": [],  # No sources for rejected questions
        "user_id": state.get("user_id"),
        "chat_history": updated_history,
    }
