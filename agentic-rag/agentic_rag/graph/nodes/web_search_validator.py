"""
Validator to check if a question is appropriate for web search.
Only allows web search for course-related technical questions.
"""

from typing import Dict, Any, Tuple
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableLambda

from dotenv import load_dotenv

from graph.chains.llm_config import create_llm, rate_limit_delay
from graph.state import GraphState

# Only load .env file if not in Docker (override=False prevents overriding existing env vars)
load_dotenv(override=False)


class WebSearchValidation(BaseModel):
    """Validate if web search is appropriate for this question."""
    
    is_valid: bool = Field(
        ...,
        description="True only if question is about current technical information related to course topics"
    )
    reason: str = Field(
        ...,
        description="Brief explanation of why web search is or isn't appropriate"
    )


llm_validator = create_llm(model="deepseek-chat", temperature=0)
structured_validator = llm_validator.with_structured_output(WebSearchValidation)

validation_message = """You are a validator for EdTech. Your job is to determine if a user's question is appropriate for web search.

CRITICAL: This is EdTech - an EDUCATIONAL PLATFORM focused on courses and learning. Web search should ONLY be used for:
- Current technical information directly related to course topics (e.g., "What's the latest React version?" for a React course)
- Recent updates to technologies covered in courses
- Breaking changes or security advisories for technologies in courses

Web search is NOT appropriate for:
- General knowledge questions
- Questions about current events, news, politics, sports, weather
- Questions about entertainment, celebrities, personal topics
- Questions that can be answered from course materials or platform knowledge base
- Questions about platform usage (these are in knowledge base)
- Any question unrelated to educational/course content

Rules:
1. Return is_valid=True ONLY if question is about current technical info related to course topics
2. Return is_valid=False for everything else (including general questions, platform questions, etc.)
3. Be strict - when in doubt, return False

Examples:
- "What's the latest React version?" -> True (current technical info for React course)
- "How do I create a course?" -> False (platform question, in knowledge base)
- "What's the weather today?" -> False (not course-related)
- "Explain React hooks" -> False (can be answered from course materials)
- "Is there a new CVE for PostgreSQL?" -> True (current security info for database course)
"""

validation_prompt = ChatPromptTemplate.from_messages(
    [("system", validation_message), ("human", "Question: {question}")]
)

base_validator = validation_prompt | structured_validator


def _rate_limited_validate(input_dict: dict):
    """Wrapper to add rate limiting to validator"""
    rate_limit_delay()
    return base_validator.invoke(input_dict)


web_search_validator = RunnableLambda(_rate_limited_validate)


def validate_web_search(question: str) -> Tuple[bool, str]:
    """
    Validate if web search is appropriate for this question.
    
    Returns:
        (is_valid, reason): Tuple of validation result and reason
    """
    try:
        result = web_search_validator.invoke({"question": question})
        return result.is_valid, result.reason
    except Exception as e:
        print(f"[VALIDATOR] Error validating question: {e}")
        # On error, be conservative and disallow web search
        return False, f"Validation error: {str(e)}"
