from typing import Any, Dict, List

from langchain_core.documents import Document

from database import fetch_user_enrollments
from ingestion import retriever
from graph.state import GraphState


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
    question = state["question"]
    user_id = state.get("user_id")
    lesson_id = state.get("lesson_id")
    chat_history = state.get("chat_history", [])
    
    # If lesson_id is provided, we only retrieve documents from that specific lesson
    if lesson_id:
        print(f"---LESSON FILTER MODE: Only retrieving documents from lesson_id={lesson_id}---")

    # Check if this is a platform usage question (excludes course recommendation)
    is_platform_question = _is_platform_question(question)
    is_course_recommendation = _is_course_recommendation_question(question)
    
    if is_course_recommendation:
        print("---DETECTED COURSE RECOMMENDATION QUESTION - USING COURSE CONTENT---")
    
    # Optimize for platform questions: enhance query + post-filter knowledge-base
    if is_platform_question:
        print("---DETECTED PLATFORM USAGE QUESTION - OPTIMIZING FOR KNOWLEDGE BASE---")
        
        # Enhance query with knowledge-base keywords to boost KB in vector search
        kb_keywords = "knowledge base guide tutorial platform guide user guide instructor guide"
        enhanced_query = f"{question} {kb_keywords}"
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
        enhanced_query = question
        
        # Enhance query with conversation history for follow-up questions
        if chat_history:
            # Get the last question and answer for context
            last_question, last_answer = chat_history[-1]
            
            # If current question is short/ambiguous, enhance with context
            # Common follow-up patterns: "give me", "show me", "what about", "how about", "tell me more"
            follow_up_keywords = [
                "give me", "show me", "what about", "how about", "tell me more",
                "example", "examples", "code", "demo", "demonstrate",
                "cho tôi", "ví dụ", "code", "mẫu"
            ]
            
            is_follow_up = any(keyword in question.lower() for keyword in follow_up_keywords)
            
            if is_follow_up or len(question.split()) < 5:
                # Enhance query with context from previous conversation
                enhanced_query = f"{last_question} {question} {last_answer[:200]}"
                print(f"---ENHANCED QUERY WITH CONVERSATION CONTEXT---")
                print(f"Original: {question}")
                print(f"Enhanced: {enhanced_query[:200]}...")
        
        documents: List[Document] = retriever.invoke(enhanced_query)

    # Filter by lesson_id if provided (priority filter - applies before user permission check)
    # When lesson_id is provided, ONLY retrieve documents from that specific lesson
    # Knowledge-base documents are excluded when filtering by lesson_id
    if lesson_id:
        lesson_filtered_documents = []
        for doc in documents:
            metadata = doc.metadata or {}
            doc_lesson_id = metadata.get("lesson_id")
            # Only include documents that match the lesson_id exactly
            # Exclude knowledge_base documents when filtering by lesson
            if doc_lesson_id == lesson_id:
                lesson_filtered_documents.append(doc)
            else:
                print(f"---DOCUMENT FILTERED: Not from lesson_id={lesson_id} (doc lesson_id={doc_lesson_id}, doc_type={metadata.get('doc_type')})---")
        documents = lesson_filtered_documents
        print(f"---LESSON FILTER RESULT: {len(documents)} documents from lesson_id={lesson_id}---")
        if len(documents) == 0:
            print(f"---WARNING: No documents found for lesson_id={lesson_id}, will trigger web search if no relevant docs---")

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
        "question": question,
        "user_id": user_id,
        "lesson_id": lesson_id,  # Preserve lesson_id in state
        "is_platform_question": is_platform_question,  # Track if this is a platform question
        "chat_history": state.get("chat_history", []),
        "regeneration_count": 0,  # Reset regeneration count for new retrieval
    }
