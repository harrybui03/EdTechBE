from typing import Literal

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableLambda

from graph.chains.llm_config import create_llm, rate_limit_delay

# Only load .env file if not in Docker (override=False prevents overriding existing env vars)
load_dotenv(override=False)


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="Route the user query to the vectorstore or websearch. Avalable options are 'vectorstore' or 'web_search'",
    )


llm = create_llm(model="deepseek-chat", temperature=0)
structured_llm_router = llm.with_structured_output(RouteQuery)

message = """You are an expert router for EdTech that decides whether a user's question should be answered using the internal vectorstore or web search.

The vectorstore contains educational content from EdTech including:
- Course overviews and descriptions
- Lesson content (text, video transcripts, attachments)
- Audio transcripts from video lessons (translated to English when available)
- Knowledge base guides (how to use the platform, create courses, enroll, etc.)
- Topics covering: programming, databases (PostgreSQL, MySQL, etc.), web development, software engineering, and various technical subjects

CRITICAL ROUTING RULES - This is an EDUCATIONAL PLATFORM, NOT a general-purpose assistant:

1. Route to **vectorstore** for ALL questions about:
   - Course content, lessons, transcripts, or educational materials
   - Technical concepts, definitions, explanations (programming, SQL, frameworks, algorithms, etc.)
   - How-to guidance, code examples, tutorials, or walkthroughs
   - Platform usage (how to create courses, enroll, publish, etc.)
   - Learning and educational topics
   - ANY question that could be answered from course materials or platform knowledge base

2. Route to **web_search** ONLY when ALL of these conditions are met:
   - Question is SPECIFICALLY about current/time-sensitive technical information (latest framework version, recent CVE, breaking changes in 2025, etc.)
   - Question is DIRECTLY related to course topics or educational content
   - Information cannot be found in course materials (e.g., "What's the latest React version?" for a React course)
   - Question is NOT about general topics (news, politics, sports, weather, entertainment, etc.)

3. STRICTLY FORBIDDEN for web_search:
   - General knowledge questions unrelated to courses
   - Questions about current events, news, politics, sports
   - Questions about weather, entertainment, celebrities
   - Personal questions or general conversation
   - Questions that can be answered from course materials or knowledge base
   - Questions about platform usage (these are in knowledge base)

4. When uncertain:
   - ALWAYS default to **vectorstore**
   - Only use web_search if question is SPECIFICALLY about current technical info related to course topics
   - If question seems unrelated to education/courses, route to vectorstore and let it handle gracefully

IMPORTANT:
- This is an EDUCATIONAL PLATFORM - stay focused on learning and courses
- Prefer vectorstore in 95% of cases - it has comprehensive educational content
- Be VERY conservative with web_search - only for current technical info related to course topics
- If question is not about courses/education/platform, route to vectorstore anyway (it will provide appropriate response)
"""
router_prompt = ChatPromptTemplate.from_messages(
    [("system", message), ("human", "{question}")]
)

base_router = router_prompt | structured_llm_router


def _rate_limited_invoke(input_dict: dict):
    """Wrapper to add rate limiting to router chain"""
    rate_limit_delay()
    return base_router.invoke(input_dict)


question_router = RunnableLambda(_rate_limited_invoke)
