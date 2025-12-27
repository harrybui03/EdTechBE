import logging
from typing import Any, Dict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from graph.state import GraphState
from graph.chains.llm_config import create_llm, rate_limit_delay
from graph.nodes.retrieve import _detect_language

logger = logging.getLogger("agentic_rag.greeting")


def _is_greeting(question: str) -> bool:
    """
    Detect simple greeting/chit-chat messages using pattern matching.
    Does not require LLM to avoid unnecessary resource consumption.
    
    Args:
        question: User's question
        
    Returns:
        True if it's a greeting/chit-chat, False otherwise
    """
    question_lower = question.strip().lower()
    
    # List of common greeting patterns
    greeting_patterns = [
        # English
        "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
        "greetings", "howdy", "what's up", "sup", "yo",
        # Vietnamese
        "xin chào", "chào", "chào bạn", "chào anh", "chào chị", "chào em",
        "xin chào bạn", "chào mừng", "hi", "hello",
        # Questions about AI itself
        "who are you", "what are you", "bạn là ai", "bạn là gì",
        "giới thiệu về bạn", "tell me about yourself",
        # Thank you
        "thank you", "thanks", "cảm ơn", "cám ơn",
        # Goodbye
        "goodbye", "bye", "see you", "tạm biệt", "chào tạm biệt",
        # Simple questions that don't require knowledge
        "how are you", "how do you do", "bạn khỏe không", "bạn thế nào",
    ]
    
    # Check if the question contains only greeting patterns (no knowledge keywords)
    for pattern in greeting_patterns:
        if pattern in question_lower:
            # Check if it's a pure greeting
            # (no knowledge keywords, technical questions, etc.)
            knowledge_keywords = [
                "what is", "what are", "how to", "explain", "define",
                "là gì", "làm sao", "như thế nào", "giải thích",
                "code", "programming", "tutorial", "example",
                "course", "lesson", "khóa học", "bài học",
            ]
            
            has_knowledge_keyword = any(kw in question_lower for kw in knowledge_keywords)
            
            # If it's just a simple greeting (short and no knowledge keywords)
            if not has_knowledge_keyword and len(question.split()) <= 5:
                return True
    
    return False


def greeting(state: GraphState) -> Dict[str, Any]:
    """
    Handle greeting/chit-chat messages by responding directly from LLM.
    Does not require RAG, web search, or other resource-intensive steps.
    
    Args:
        state: Current state of the graph
        
    Returns:
        Dictionary containing generation and necessary information
    """
    print("---GREETING/CHIT-CHAT---")
    logger.info("---GREETING/CHIT-CHAT---")
    question = state["question"]
    chat_history = state.get("chat_history", [])
    
    # Create simple prompt for greeting
    greeting_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a friendly and helpful AI assistant for EdTech.
When users greet you or engage in small talk, respond warmly and briefly.
Keep responses concise (1-2 sentences) and natural.
If they ask about yourself, briefly introduce yourself as an AI assistant for EdTech that helps with course-related questions.
When referring to the platform, use "EdTech" rather than generic terms.
If they thank you, respond politely.
If they say goodbye, respond appropriately."""),
        ("human", "{question}")
    ])
    
    # Create simple chain for greeting (no document context needed)
    llm = create_llm(model="deepseek-chat", temperature=0.7)  # Higher temperature for more natural responses
    greeting_chain = greeting_prompt | llm | StrOutputParser()
    
    # Add conversation context if available
    if chat_history:
        # Get the most recent messages for context
        recent_context = "\n".join([
            f"User: {q}\nAssistant: {a}" 
            for q, a in chat_history[-3:]  # Only take the last 3 messages
        ])
        enhanced_question = f"Previous conversation:\n{recent_context}\n\nCurrent message: {question}"
        rate_limit_delay()  # Call rate limit before invoke
        generation = greeting_chain.invoke({"question": enhanced_question})
    else:
        rate_limit_delay()  # Call rate limit before invoke
        generation = greeting_chain.invoke({"question": question})
    
    # Update chat history
    updated_history = list(chat_history) if chat_history else []
    updated_history.append((question, generation))
    
    # Detect and preserve original language
    original_language = state.get("original_language") or _detect_language(question)
    
    return {
        "generation": generation,
        "documents": [],  # No documents for greeting
        "question": question,
        "sources": [],  # No sources for greeting
        "user_id": state.get("user_id"),
        "chat_history": updated_history,
        "original_language": original_language,  # Preserve original language
    }

