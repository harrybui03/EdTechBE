from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableLambda, RunnableSequence

from graph.chains.llm_config import create_llm, rate_limit_delay

# Only load .env file if not in Docker (override=False prevents overriding existing env vars)
load_dotenv(override=False)


class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generated answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'."
    )


llm = create_llm(model="deepseek-chat", temperature=0)

structured_llm_grader = llm.with_structured_output(GradeHallucinations)

message = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", message),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

base_grader: RunnableSequence = hallucination_prompt | structured_llm_grader


def _rate_limited_invoke(input_dict: dict):
    """Wrapper to add rate limiting to hallucination grader chain"""
    rate_limit_delay()
    return base_grader.invoke(input_dict)


hallucination_grader = RunnableLambda(_rate_limited_invoke)
