from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableLambda, RunnableSequence

from graph.chains.llm_config import create_llm, rate_limit_delay

# Only load .env file if not in Docker (override=False prevents overriding existing env vars)
load_dotenv(override=False)


class GradeAnswer(BaseModel):
    """Binary score for assessing the answer is relevant to the question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'."
    )


llm = create_llm(model="deepseek-chat", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeAnswer)

message = """You are a grader assessing whether an answer addresses / resolves a question \n 
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", message),
        ("human", "User question: {question} \n\n LLM generated answer: {generation}"),
    ]
)

base_grader: RunnableSequence = answer_prompt | structured_llm_grader


def _rate_limited_invoke(input_dict: dict):
    """Wrapper to add rate limiting to answer grader chain"""
    rate_limit_delay()
    return base_grader.invoke(input_dict)


answer_grader = RunnableLambda(_rate_limited_invoke)
