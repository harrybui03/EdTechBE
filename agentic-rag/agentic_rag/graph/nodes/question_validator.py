"""
Early validator to reject questions unrelated to courses/education.
This prevents unnecessary API calls for questions like "how is the weather today".
"""

from typing import Tuple
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableLambda

from dotenv import load_dotenv

from graph.chains.llm_config import create_llm, rate_limit_delay

# Only load .env file if not in Docker (override=False prevents overriding existing env vars)
load_dotenv(override=False)


class QuestionValidation(BaseModel):
    """Validate if question is related to courses/education."""
    
    is_related: bool = Field(
        ...,
        description="True if question is related to courses, education, platform usage, or technical topics"
    )
    reason: str = Field(
        ...,
        description="Brief explanation of why question is or isn't related"
    )


llm_validator = create_llm(model="deepseek-chat", temperature=0)
structured_validator = llm_validator.with_structured_output(QuestionValidation)

validation_message = """You are a validator for EdTech. Your job is to determine if a user's question is related to courses, education, or platform usage.

CRITICAL: This is EdTech - an EDUCATIONAL PLATFORM. Questions are considered RELATED if they are about:
- Course content, lessons, or educational materials
- Technical topics (programming, databases, frameworks, etc.)
- Platform usage (how to create courses, enroll, publish, etc.)
- Learning and education topics
- Questions that could be answered from course materials or knowledge base

Questions are considered UNRELATED if they are about:
- Weather, current events, news
- Politics, sports, entertainment
- Personal questions unrelated to learning
- General knowledge questions not covered in courses
- Questions clearly outside educational scope

Rules:
1. Return is_related=True if question is about courses, education, platform, or technical topics
2. Return is_related=False for weather, news, sports, entertainment, etc.
3. Be strict - when in doubt about unrelated topics, return False

Examples:
- "How is the weather today?" -> False (not related to courses)
- "How do I create a course?" -> True (platform usage)
- "What is React?" -> True (technical topic, could be in courses)
- "Who won the game?" -> False (sports, not related)
- "Explain SQL joins" -> True (technical topic, likely in courses)
- "What's the latest news?" -> False (news, not related)
"""

validation_prompt = ChatPromptTemplate.from_messages(
    [("system", validation_message), ("human", "Question: {question}")]
)

base_validator = validation_prompt | structured_validator


def _rate_limited_validate(input_dict: dict):
    """Wrapper to add rate limiting to validator"""
    rate_limit_delay()
    return base_validator.invoke(input_dict)


question_validator = RunnableLambda(_rate_limited_validate)


def validate_question(question: str) -> Tuple[bool, str]:
    """
    Validate if question is related to courses/education.
    
    Returns:
        (is_related, reason): Tuple of validation result and reason
    """
    try:
        result = question_validator.invoke({"question": question})
        return result.is_related, result.reason
    except Exception as e:
        print(f"[VALIDATOR] Error validating question: {e}")
        # On error, be conservative and allow (let normal flow handle it)
        return True, f"Validation error: {str(e)}"


def _is_unrelated_question_simple(question: str) -> bool:
    """
    Simple pattern-based check for obviously unrelated questions.
    This is a fast check before using LLM validator.
    
    Returns:
        True if question is clearly unrelated (weather, news, etc.)
    """
    question_lower = question.lower().strip()
    
    # Obvious unrelated patterns
    unrelated_patterns = [
        # Weather
        "weather", "temperature", "rain", "snow", "sunny", "cloudy",
        "thời tiết", "mưa", "nắng",
        
        # News/Current events
        "news", "latest news", "what happened", "current events",
        "tin tức", "tin mới",
        
        # Sports
        "game", "match", "score", "team", "player", "sport",
        "trận đấu", "đội", "cầu thủ",
        
        # Entertainment
        "movie", "film", "celebrity", "actor", "singer",
        "phim", "diễn viên", "ca sĩ",
        
        # Personal (unrelated to learning)
        "how are you", "what's your name", "tell me a joke",
        "bạn khỏe không", "tên bạn là gì", "kể chuyện cười",
    ]
    
    # Check if question starts with or contains these patterns
    for pattern in unrelated_patterns:
        if pattern in question_lower:
            # But exclude if it's about course content (e.g., "weather API in course")
            course_keywords = ["course", "lesson", "tutorial", "learn", "khóa học", "bài học"]
            has_course_context = any(kw in question_lower for kw in course_keywords)
            
            if not has_course_context:
                return True
    
    return False
