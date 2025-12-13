"""
Batch retrieval grader to grade multiple documents in one API call.
Reduces API calls from N (one per document) to 1 (batch).
"""

from typing import List
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableLambda

from graph.chains.llm_config import create_llm, rate_limit_delay

# Only load .env file if not in Docker (override=False prevents overriding existing env vars)
load_dotenv(override=False)

llm = create_llm(model="deepseek-chat", temperature=0)


class BatchGradeDocuments(BaseModel):
    """Batch grading result for multiple documents"""
    
    relevant_document_indices: List[int] = Field(
        ...,
        description="List of indices (0-based) of documents that are relevant to the question. Empty list if none are relevant."
    )
    reasoning: str = Field(
        ...,
        description="Brief explanation of which documents are relevant and why"
    )


structured_llm_grader = llm.with_structured_output(BatchGradeDocuments)

message = """You are a grader assessing relevance of retrieved documents to a user question.

You will receive multiple documents numbered from 0. For each document, determine if it is relevant to the question.

A document is relevant if:
- It contains keywords or semantic meaning related to the question
- It provides information that could help answer the question
- It is about topics covered in the question

Return the indices (0-based) of ALL relevant documents. If no documents are relevant, return an empty list.

Important:
- Be strict: only include documents that are truly relevant
- Consider semantic meaning, not just keyword matching
- If question is about courses/education, documents about courses are relevant
- If question is unrelated to courses, most documents will be irrelevant
"""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", message),
        ("human", """User question: {question}

Documents to grade (numbered from 0):

{documents}

Return the indices of relevant documents (0-based). If none are relevant, return empty list."""),
    ]
)

base_grader = grade_prompt | structured_llm_grader


def _rate_limited_invoke(input_dict: dict):
    """Wrapper to add rate limiting to batch retrieval grader chain"""
    rate_limit_delay()
    return base_grader.invoke(input_dict)


batch_retrieval_grader = RunnableLambda(_rate_limited_invoke)
