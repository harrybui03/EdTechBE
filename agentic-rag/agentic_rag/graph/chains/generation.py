from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from graph.chains.llm_config import create_llm, rate_limit_delay

# Only load .env file if not in Docker (override=False prevents overriding existing env vars)
load_dotenv(override=False)

llm = create_llm(model="deepseek-chat", temperature=0)

# Custom prompt template for balanced, comprehensive answers
system_message = """You are an expert AI assistant for EdTech. Provide clear, helpful answers to students' questions about course content, technical topics, and platform usage.

IMPORTANT: When referring to the platform, use "EdTech" rather than generic terms like "web elearning" or "e-learning website".

GUIDELINES:
- Provide complete answers that address all aspects of the question
- Match answer length to question complexity (simple = concise, complex = detailed)
- Base answers primarily on the provided context/documents
- Start with a direct answer, then add explanations and examples as needed
- Use simple lists (-) and numbered lists (1., 2., 3.) when helpful for clarity
- Avoid markdown formatting like **bold** or ## headers - use plain text instead
- For code examples, use plain text code blocks when necessary
- Focus on helping students learn and understand

REMEMBER: Prioritize accuracy, ground answers in provided documents, balance completeness with conciseness. You can use simple lists and numbered lists, but avoid markdown formatting like **bold** or ## headers."""

# Short prompt template for platform/knowledge-base questions (concise like knowledge base docs)
system_message_platform = """You are an expert AI assistant for EdTech. Answer questions about platform usage concisely and clearly, similar to knowledge base documentation.

IMPORTANT: When referring to the platform, use "EdTech".

GUIDELINES:
- Provide direct, concise answers (similar to knowledge base documentation style)
- Be clear and practical, avoid lengthy explanations
- Base answers on the provided context/documents
- Use simple lists (-) and numbered lists (1., 2., 3.) when helpful for clarity
- Avoid markdown formatting like **bold** or ## headers - use plain text instead
- Keep answers focused and to the point

REMEMBER: For platform questions, users want quick, clear answers like in documentation - be concise and practical. You can use simple lists and numbered lists, but avoid markdown formatting like **bold** or ## headers."""

human_template = """Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Provide a comprehensive, detailed answer that thoroughly addresses the question using the context provided above. You can use simple lists (-) and numbered lists (1., 2., 3.) when helpful, but avoid markdown formatting like **bold** or ## headers."""

human_template_platform = """Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Provide a concise, clear answer similar to knowledge base documentation style. You can use simple lists (-) and numbered lists (1., 2., 3.) when helpful, but avoid markdown formatting like **bold** or ## headers."""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_message),
    ("human", human_template)
])

prompt_platform = ChatPromptTemplate.from_messages([
    ("system", system_message_platform),
    ("human", human_template_platform)
])

base_chain = prompt | llm | StrOutputParser()
base_chain_platform = prompt_platform | llm | StrOutputParser()


def _rate_limited_invoke(input_dict: dict):
    """Wrapper to add rate limiting to generation chain"""
    rate_limit_delay()
    return base_chain.invoke(input_dict)


def _rate_limited_invoke_platform(input_dict: dict):
    """Wrapper to add rate limiting to platform generation chain"""
    rate_limit_delay()
    return base_chain_platform.invoke(input_dict)


generation_chain = RunnableLambda(_rate_limited_invoke)
generation_chain_platform = RunnableLambda(_rate_limited_invoke_platform)
