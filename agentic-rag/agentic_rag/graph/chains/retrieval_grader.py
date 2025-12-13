from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableLambda

from graph.chains.llm_config import create_llm, rate_limit_delay

# Only load .env file if not in Docker (override=False prevents overriding existing env vars)
load_dotenv(override=False)

llm = create_llm(model="deepseek-chat", temperature=0)


class GradeDocuments(BaseModel):
    """Binary score for the relevance check of retrieved documents"""

    binary_score: str = Field(
        description="Documents are relevant to the question? 'yes' or 'no'",
    )


structured_llm_grader = llm.with_structured_output(GradeDocuments)

message = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", message),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

base_grader = grade_prompt | structured_llm_grader


def _rate_limited_invoke(input_dict: dict):
    """Wrapper to add rate limiting to retrieval grader chain"""
    rate_limit_delay()
    return base_grader.invoke(input_dict)


retrieval_grader = RunnableLambda(_rate_limited_invoke)
