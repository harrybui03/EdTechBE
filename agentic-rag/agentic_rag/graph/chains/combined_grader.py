"""
Combined grader that checks both hallucination and answer relevance in one API call.
Reduces API calls from 2 (hallucination + answer) to 1 (combined).
"""

from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableLambda

from graph.chains.llm_config import create_llm, rate_limit_delay

# Only load .env file if not in Docker (override=False prevents overriding existing env vars)
load_dotenv(override=False)

llm = create_llm(model="deepseek-chat", temperature=0)


class CombinedGrade(BaseModel):
    """Combined grading result for hallucination and answer relevance"""
    
    is_grounded: bool = Field(
        ...,
        description="True if the answer is grounded in/supported by the documents"
    )
    addresses_question: bool = Field(
        ...,
        description="True if the answer addresses/resolves the user question"
    )
    reasoning: str = Field(
        ...,
        description="Brief explanation of the grading decisions"
    )


structured_llm_grader = llm.with_structured_output(CombinedGrade)

message = """You are a grader assessing an LLM-generated answer on two criteria:

1. **Grounding**: Is the answer grounded in/supported by the provided documents?
   - Check if facts, claims, and information in the answer can be found in the documents
   - Answer should not contain information not present in documents (hallucinations)
   - Answer can synthesize information from multiple documents

2. **Relevance**: Does the answer address/resolve the user's question?
   - Check if the answer actually answers what the user asked
   - Answer should be relevant and useful for the question
   - Answer should not be off-topic or unhelpful

Return both assessments:
- is_grounded: True if answer is supported by documents
- addresses_question: True if answer addresses the question

Important:
- Be strict: both criteria should be met for a good answer
- If answer is not grounded, it's not useful even if it seems relevant
- If answer doesn't address the question, it's not useful even if grounded
"""
combined_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", message),
        ("human", """User question: {question}

Retrieved documents:
{documents}

LLM-generated answer:
{generation}

Assess both:
1. Is the answer grounded in the documents? (is_grounded)
2. Does the answer address the question? (addresses_question)"""),
    ]
)

base_grader = combined_prompt | structured_llm_grader


def _rate_limited_invoke(input_dict: dict):
    """Wrapper to add rate limiting to combined grader chain"""
    rate_limit_delay()
    return base_grader.invoke(input_dict)


combined_grader = RunnableLambda(_rate_limited_invoke)
