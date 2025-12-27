import logging
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from database import fetch_user_enrollments
from ingestion import retriever, vectorstore
from graph.state import GraphState
from graph.chains.llm_config import create_llm, rate_limit_delay

logger = logging.getLogger("agentic_rag.retrieve")


def _is_course_recommendation_question(question: str) -> bool:
    """
    Detect if question is about course recommendation (which course to take).
    These questions should use course content, NOT knowledge-base.
    """
    question_lower = question.lower()
    
    # Course recommendation patterns
    recommendation_patterns = [
        "what course", "which course", "what courses",
        "course should i", "courses should i", "course to take", "courses to take",
        "course to learn", "courses to learn", "course for", "courses for",
        "recommend course", "recommend courses", "suggest course", "suggest courses",
        "best course", "best courses", "good course", "good courses",
        "course to become", "courses to become", "course to be", "courses to be",
        "khóa học nào", "khóa học để", "nên học khóa học nào",
    ]
    
    return any(pattern in question_lower for pattern in recommendation_patterns)


def _detect_language(text: str) -> str:
    """
    Simple language detection: check if text contains Vietnamese characters.
    Returns 'vi' for Vietnamese, 'en' for English (default).
    """
    vietnamese_chars = set('àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ')
    text_lower = text.lower()
    # Check if text contains Vietnamese characters
    if any(char in vietnamese_chars for char in text_lower):
        return 'vi'
    # Check for common Vietnamese words
    vietnamese_words = ['tôi', 'bạn', 'của', 'là', 'và', 'với', 'cho', 'được', 'có', 'không', 'nào', 'gì', 'sao', 'thế', 'cách', 'để', 'thanh', 'toán']
    words = text_lower.split()
    vietnamese_word_count = sum(1 for word in words if word in vietnamese_words)
    if vietnamese_word_count >= 1:  # At least 1 Vietnamese word
        return 'vi'
    return 'en'


def _translate_to_english(question: str) -> str:
    """
    Translate non-English question to English for better vector search.
    Uses LLM for translation to ensure quality.
    
    Args:
        question: Original question in any language
        
    Returns:
        Translated question in English
    """
    # Create translation prompt
    translation_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a professional translator. Translate the user's question to English while preserving the meaning and intent. 
Return ONLY the English translation, no explanations or additional text."""),
        ("human", "{question}")
    ])
    
    llm = create_llm(model="deepseek-chat", temperature=0)
    translation_chain = translation_prompt | llm | StrOutputParser()
    
    try:
        rate_limit_delay()
        translated = translation_chain.invoke({"question": question})
        print(f"---TRANSLATED QUERY: '{question}' -> '{translated}'---")
        logger.info(f"---TRANSLATED QUERY: '{question}' -> '{translated}'---")
        return translated.strip()
    except Exception as e:
        print(f"---TRANSLATION ERROR: {e}, USING ORIGINAL QUESTION---")
        logger.warning(f"---TRANSLATION ERROR: {e}, USING ORIGINAL QUESTION---", exc_info=True)
        return question  # Fallback to original if translation fails


def _is_platform_question(question: str) -> bool:
    """
    Detect if question is about platform usage (how to use the platform).
    These questions should prioritize knowledge-base documents.
    
    Excludes course recommendation questions which should use course content.
    """
    question_lower = question.lower()
    
    # Exclude course recommendation questions first
    if _is_course_recommendation_question(question):
        return False
    
    # Platform usage keywords (must be about HOW TO USE the platform)
    platform_keywords = [
        # How-to patterns (must be about platform usage, not course content)
        "how to", "how do i", "how can i", "how does",
        # Platform actions (instructor)
        "create a course", "create course", "publish a course", "publish course",
        "how to create", "how to publish", "how to manage",
        # Platform actions (user) - but only HOW TO enroll, not WHAT to enroll
        "how to enroll", "how do i enroll", "how can i enroll",
        "how to access", "how to use dashboard",
        # Platform features
        "dashboard", "platform", "manage course", "manage lesson",
        "add lesson", "add chapter", "course structure",
        "landing page", "course pricing", "course settings",
        "instructor guide", "user guide", "platform guide",
        # Vietnamese
        "làm sao", "làm thế nào", "cách",
        "tạo khóa học", "publish khóa học",
        "cách đăng ký", "làm sao đăng ký",  # HOW to enroll
        "quản lý khóa học", "quản lý bài học",
    ]
    
    # Check if question contains platform keywords
    has_platform_keyword = any(keyword in question_lower for keyword in platform_keywords)
    
    # Additional check: "enroll" alone is ambiguous - need "how to enroll"
    if "enroll" in question_lower or "enrollment" in question_lower or "enrolling" in question_lower:
        # Only consider platform question if it's about HOW to enroll
        if "how" in question_lower or "cách" in question_lower or "làm sao" in question_lower:
            return True
        # Otherwise, it's likely a course recommendation question
        return False
    
    return has_platform_keyword


def retrieve(state: GraphState) -> Dict[str, Any]:
    """
    Retrieve documents from the retriever.
    Enhances query with conversation history for better context-aware retrieval.
    Prioritizes knowledge-base documents for platform usage questions.

    Args:
        state: The current state of the graph.

    Returns:
        A dictionary containing the retrieved documents and the question
    """
    print("---RETRIEVE---")
    logger.info("---RETRIEVE---")
    question = state["question"]
    user_id = state.get("user_id")
    lesson_id = state.get("lesson_id")
    chat_history = state.get("chat_history", [])
    
    # Detect original language and translate to English for better vector search
    original_language = _detect_language(question)
    search_query = question  # Default to original question
    
    if original_language != 'en':
        print(f"---DETECTED NON-ENGLISH QUESTION (language: {original_language}), TRANSLATING TO ENGLISH---")
        logger.info(f"---DETECTED NON-ENGLISH QUESTION (language: {original_language}), TRANSLATING TO ENGLISH---")
        search_query = _translate_to_english(question)
    else:
        print("---QUESTION IS IN ENGLISH, NO TRANSLATION NEEDED---")
        logger.info("---QUESTION IS IN ENGLISH, NO TRANSLATION NEEDED---")
    
    # If lesson_id is provided, we only retrieve documents from that specific lesson
    if lesson_id:
        msg = f"---LESSON FILTER MODE: Only retrieving documents from lesson_id={lesson_id}---"
        print(msg)
        logger.info(msg)

    # Check if this is a platform usage question (excludes course recommendation)
    # Use original question for detection, but search_query (translated) for retrieval
    is_platform_question = _is_platform_question(question)
    is_course_recommendation = _is_course_recommendation_question(question)
    
    if is_course_recommendation:
        print("---DETECTED COURSE RECOMMENDATION QUESTION - USING COURSE CONTENT---")
    
    # Optimize for platform questions: enhance query + post-filter knowledge-base
    if is_platform_question:
        print("---DETECTED PLATFORM USAGE QUESTION - OPTIMIZING FOR KNOWLEDGE BASE---")
        
        # Enhance query with knowledge-base keywords to boost KB in vector search
        # Use translated query for better search results
        kb_keywords = "knowledge base guide tutorial platform guide user guide instructor guide"
        enhanced_query = f"{search_query} {kb_keywords}"
        print(f"---ENHANCED QUERY FOR KB: {enhanced_query[:150]}...---")
        
        # Retrieve documents (enhanced query helps KB rank higher)
        all_documents = retriever.invoke(enhanced_query)
        
        # Post-filter: Separate knowledge-base and other documents
        kb_documents = []
        other_documents = []
        for doc in all_documents:
            metadata = doc.metadata or {}
            if metadata.get("doc_type") == "knowledge_base":
                kb_documents.append(doc)
            else:
                other_documents.append(doc)
        
        # Optimization: Prioritize KB, limit total documents to reduce token usage
        if len(kb_documents) >= 3:
            # Use only KB docs if we have enough (max 5 for cost optimization)
            documents = kb_documents[:5]
            print(f"---OPTIMIZED: Using {len(documents)} KB docs only (post-filtered)---")
        elif len(kb_documents) > 0:
            # If we have some KB docs, prioritize them + add top 2 course docs for context
            documents = kb_documents + other_documents[:2]
            print(f"---OPTIMIZED: {len(kb_documents)} KB docs + {len(other_documents[:2])} course docs (post-filtered)---")
        else:
            # Fallback: No KB docs found, use all docs but log warning
            documents = all_documents
            print(f"---WARNING: No KB docs found for platform question, using all {len(documents)} docs---")
    else:
        # Normal retrieval for course content questions (including course recommendations)
        # Use translated query for better search results
        enhanced_query = search_query
        
        # When filtering by lesson_id, enhance query to boost transcript and lesson documents
        # This helps find lesson-specific content even if query is generic
        if lesson_id:
            # Enhance query with lesson-specific keywords to boost transcript/lesson documents
            # Transcript documents contain "Audio Transcript:" prefix, so we add keywords that match
            lesson_boost_keywords = "lesson transcript audio video content course material"
            enhanced_query = f"{search_query} {lesson_boost_keywords}"
            print(f"---ENHANCED QUERY FOR LESSON: {enhanced_query[:200]}...---")
        
        # Enhance query with conversation history for follow-up questions
        if chat_history:
            # Get the last question and answer for context
            last_question, last_answer = chat_history[-1]
            
            # Translate last question if needed for better context
            last_question_lang = _detect_language(last_question)
            if last_question_lang != 'en':
                last_question_translated = _translate_to_english(last_question)
            else:
                last_question_translated = last_question
            
            # If current question is short/ambiguous, enhance with context
            # Common follow-up patterns: "give me", "show me", "what about", "how about", "tell me more"
            follow_up_keywords = [
                "give me", "show me", "what about", "how about", "tell me more",
                "example", "examples", "code", "demo", "demonstrate",
                "cho tôi", "ví dụ", "code", "mẫu"
            ]
            
            is_follow_up = any(keyword in question.lower() for keyword in follow_up_keywords)
            
            if is_follow_up or len(question.split()) < 5:
                # Enhance query with context from previous conversation (use translated versions)
                enhanced_query = f"{last_question_translated} {enhanced_query} {last_answer[:200]}"
                print(f"---ENHANCED QUERY WITH CONVERSATION CONTEXT---")
                print(f"Original: {question}")
                print(f"Enhanced: {enhanced_query[:200]}...")
        
        # When filtering by lesson_id, retrieve more documents to increase chance of finding lesson content
        # Default k=7 might not include lesson documents if they rank lower
        if lesson_id:
            # Use higher k when filtering by lesson_id to ensure we get lesson documents
            # Increased to 150 to better find transcript chunks which might rank lower
            high_k_retriever = vectorstore.as_retriever(search_kwargs={"k": 150})
            documents: List[Document] = high_k_retriever.invoke(enhanced_query)
            print(f"---RETRIEVED {len(documents)} documents (k=150 for lesson_id filter)---")
        else:
            documents: List[Document] = retriever.invoke(enhanced_query)
            print(f"---RETRIEVED {len(documents)} documents (default k=7)---")

    # Filter by lesson_id if provided (priority filter - applies before user permission check)
    # When lesson_id is provided, ONLY retrieve documents from that specific lesson
    # Knowledge-base documents are excluded when filtering by lesson_id
    if lesson_id:
        # Normalize lesson_id to string for consistent comparison
        lesson_id_str = str(lesson_id).strip() if lesson_id else None
        lesson_filtered_documents = []
        
        # Debug: log all retrieved documents before filtering
        print(f"---DEBUG: Before filtering, retrieved {len(documents)} documents---")
        
        # Count documents by type and lesson_id
        doc_type_counts = {}
        transcript_count = 0
        matching_lesson_count = 0
        
        for idx, doc in enumerate(documents, 1):
            metadata = doc.metadata or {}
            doc_lesson_id = metadata.get("lesson_id")
            doc_type = metadata.get("doc_type")
            
            # Count by type
            doc_type_counts[doc_type] = doc_type_counts.get(doc_type, 0) + 1
            
            # Count transcript documents
            if doc_type == "transcript":
                transcript_count += 1
                if str(doc_lesson_id).strip() == lesson_id_str:
                    matching_lesson_count += 1
                    print(f"---DEBUG TRANSCRIPT FOUND: doc_id={metadata.get('document_id')}, lesson_id={doc_lesson_id}---")
            
            # Log first 10 for detailed debugging
            if idx <= 10:
                print(f"---DEBUG DOC {idx}: doc_type={doc_type}, lesson_id={doc_lesson_id}, doc_id={metadata.get('document_id')}---")
        
        print(f"---DEBUG SUMMARY: doc_types={doc_type_counts}, transcript_count={transcript_count}, matching_transcript_count={matching_lesson_count}---")
        
        for doc in documents:
            metadata = doc.metadata or {}
            doc_lesson_id = metadata.get("lesson_id")
            # Normalize doc_lesson_id to string for comparison
            doc_lesson_id_str = str(doc_lesson_id).strip() if doc_lesson_id else None
            
            # Only include documents that match the lesson_id exactly
            # Exclude knowledge_base documents when filtering by lesson
            if doc_lesson_id_str == lesson_id_str:
                lesson_filtered_documents.append(doc)
            else:
                print(f"---DOCUMENT FILTERED: Not from lesson_id={lesson_id_str} (doc lesson_id={doc_lesson_id_str}, doc_type={metadata.get('doc_type')})---")
        documents = lesson_filtered_documents
        print(f"---LESSON FILTER RESULT: {len(documents)} documents from lesson_id={lesson_id_str}---")
        if len(documents) == 0:
            print(f"---WARNING: No documents found for lesson_id={lesson_id_str}, will trigger web search if no relevant docs---")
            print(f"---DEBUG: Check if documents were ingested for this lesson_id---")
            print(f"---DEBUG: Consider increasing k or checking vector similarity scores---")

    # Filter by user permissions (enrollment check)
    if user_id:
        allowed_courses = fetch_user_enrollments(user_id)
        filtered_documents = []
        for doc in documents:
            metadata = doc.metadata or {}
            requires_enrollment = metadata.get("requires_enrollment", False)
            doc_course_id = metadata.get("course_id")
            if not requires_enrollment:
                filtered_documents.append(doc)
            elif doc_course_id and doc_course_id in allowed_courses:
                filtered_documents.append(doc)
            else:
                print("---DOCUMENT FILTERED: USER LACKS ACCESS---")
        documents = filtered_documents

    return {
        "documents": documents,
        "question": question,  # Keep original question for response generation
        "user_id": user_id,
        "lesson_id": lesson_id,  # Preserve lesson_id in state
        "is_platform_question": is_platform_question,  # Track if this is a platform question
        "chat_history": state.get("chat_history", []),
        "regeneration_count": 0,  # Reset regeneration count for new retrieval
        "original_language": original_language,  # Store original language for response generation
    }
