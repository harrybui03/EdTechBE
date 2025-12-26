import logging
from typing import Any, Dict, List

from langchain_core.documents import Document

from database import fetch_course_structure
from graph.chains.generation import generation_chain, generation_chain_platform
from graph.state import GraphState

logger = logging.getLogger("agentic_rag.generate")

SOURCE_KEYS = [
    "document_id",
    "doc_type",
    "course_id",
    "course_title",
    "chapter_id",
    "chapter_title",
    "lesson_id",
    "lesson_title",
    "requires_enrollment",
    "tags",
    "language",
    "course_skill_level",
    "chapter_summary",
    "last_modified",
]


def _build_context(documents: List[Document]) -> str:
    context_chunks = []
    for doc in documents:
        metadata = doc.metadata or {}
        header_parts = []
        if metadata.get("course_title"):
            header_parts.append(f"Course: {metadata['course_title']}")
        if metadata.get("chapter_title"):
            header_parts.append(f"Chapter: {metadata['chapter_title']}")
        if metadata.get("lesson_title"):
            header_parts.append(f"Lesson: {metadata['lesson_title']}")
        header = " • ".join(header_parts)
        chunk = "\n\n".join(filter(None, [header, doc.page_content]))
        context_chunks.append(chunk.strip())
    return "\n\n-----\n\n".join(context_chunks)


def _extract_sources(documents: List[Document]) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    for idx, doc in enumerate(documents):
        metadata = doc.metadata or {}
        source_entry = {"rank": idx + 1}
        for key in SOURCE_KEYS:
            if key in metadata and metadata[key] is not None:
                source_entry[key] = metadata[key]
        if "distance" in metadata:
            source_entry["distance"] = metadata["distance"]
        sources.append(source_entry)
    return sources


def _is_roadmap_question(question: str) -> bool:
    """Check if the question is about roadmap/learning path"""
    roadmap_keywords = [
        # English keywords
        "roadmap",
        "learning path",
        "learning journey",
        "study plan",
        "study guide",
        "course structure",
        "course outline",
        "course content",
        "course plan",
        "program structure",
        "syllabus",
        "curriculum",
        "outline",
        "chapters",
        "lessons order",
        "lesson sequence",
        "course sequence",
        "what to learn",
        "how to learn",
        "where to start",
        "where should i start",
        "what should i learn",
        "order of lessons",
        "order of chapters",
        "lesson order",
        "chapter order",
        "course flow",
        "learning flow",
        "study sequence",
        "learning sequence",
        "course progression",
        "learning progression",
        "course path",
        "study path",
        # Vietnamese keywords
        "lộ trình",
        "lo trinh",
        "học như thế nào",
        "hoc nhu the nao",
        "bắt đầu từ đâu",
        "bat dau tu dau",
        "thứ tự học",
        "thu tu hoc",
        "cấu trúc khóa học",
        "cau truc khoa hoc",
        "chương trình học",
        "chuong trinh hoc",
    ]
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in roadmap_keywords)


def _extract_course_id_from_documents(documents: List[Document]) -> str | None:
    """Extract course_id from documents if available"""
    for doc in documents:
        metadata = doc.metadata or {}
        course_id = metadata.get("course_id")
        if course_id:
            return str(course_id)
    return None


def _format_roadmap(course_structure: Dict[str, object]) -> str:
    """Format course structure into a well-formatted roadmap text"""
    lines = []
    
    course_title = course_structure.get("course_title", "Unknown Course")
    course_description = course_structure.get("course_description", "")
    skill_level = course_structure.get("course_skill_level", "")
    target_audience = course_structure.get("course_target_audience", "")
    
    lines.append(f"# 📚 {course_title}")
    if course_description:
        lines.append(f"\n{course_description}")
    if skill_level:
        lines.append(f"\n**Skill Level:** {skill_level}")
    if target_audience:
        lines.append(f"**Target Audience:** {target_audience}")
    
    chapters = course_structure.get("chapters", [])
    if not chapters:
        lines.append("\n\n⚠️ No chapters found in this course.")
        return "\n".join(lines)
    
    lines.append("\n\n## 📖 Course Structure\n")
    
    for chapter_idx, chapter in enumerate(chapters, start=1):
        chapter_title = chapter.get("chapter_title", "Untitled Chapter")
        chapter_summary = chapter.get("chapter_summary", "")
        position = chapter.get("position")
        
        chapter_header = f"### Chapter {chapter_idx}"
        if position is not None:
            chapter_header += f" (Position: {position})"
        chapter_header += f": {chapter_title}"
        lines.append(chapter_header)
        
        if chapter_summary:
            lines.append(f"\n{chapter_summary}\n")
        
        lessons = chapter.get("lessons", [])
        if lessons:
            lines.append("**Lessons:**")
            for lesson_idx, lesson in enumerate(lessons, start=1):
                lesson_title = lesson.get("lesson_title", "Untitled Lesson")
                lesson_position = lesson.get("position")
                has_video = lesson.get("has_video", False)
                has_file = lesson.get("has_file", False)
                
                lesson_line = f"  {chapter_idx}.{lesson_idx}. {lesson_title}"
                if lesson_position is not None:
                    lesson_line += f" (Position: {lesson_position})"
                
                badges = []
                if has_video:
                    badges.append("🎥 Video")
                if has_file:
                    badges.append("📎 File")
                if badges:
                    lesson_line += f" [{', '.join(badges)}]"
                
                lines.append(lesson_line)
        else:
            lines.append("  *No lessons in this chapter*")
        
        lines.append("")  # Empty line between chapters
    
    return "\n".join(lines)


def _build_conversation_context(chat_history: List[tuple[str, str]] | None) -> str:
    """Build conversation context from chat history"""
    if not chat_history:
        return ""
    
    # Only include last 5 exchanges to avoid context overflow
    recent_history = chat_history[-5:]
    
    context_parts = [
        "## CONVERSATION HISTORY (Use this to understand context for follow-up questions):",
        "IMPORTANT: The current question may refer to previous questions/answers. Use the conversation history to understand what the user is asking about.",
        ""
    ]
    
    for idx, (prev_question, prev_answer) in enumerate(recent_history, start=1):
        context_parts.append(f"**Previous Question {idx}:** {prev_question}")
        # Include full answer for better context, but limit length
        answer_preview = prev_answer[:500] + "..." if len(prev_answer) > 500 else prev_answer
        context_parts.append(f"**Previous Answer {idx}:** {answer_preview}")
        context_parts.append("")  # Empty line between exchanges
    
    context_parts.append("---")
    context_parts.append("Current Question: (see below)")
    context_parts.append("")
    
    return "\n".join(context_parts)


def generate(state: GraphState) -> Dict[str, Any]:
    """
    Generate a response to the user question.
    If the question is about roadmap, it will fetch and format course structure.
    Includes conversation history for context-aware responses.

    Args:
        state (dict): The current state of the graph.

    Returns:
        state (dict): A dictionary containing the generated response and the question
    """
    print("---GENERATE---")
    logger.info("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    chat_history = state.get("chat_history", [])
    regeneration_count = state.get("regeneration_count") or 0
    
    # Increment regeneration count if this is a regeneration attempt
    # (regeneration happens when coming back from grade_generation_grounded_in_documents_and_question with "not_supported")
    # Check if there's already a generation in state, which means this is a retry
    if state.get("generation") and regeneration_count == 0:
        # First regeneration attempt
        regeneration_count = 1
        msg = f"---REGENERATION ATTEMPT #1---"
        print(msg)
        logger.info(msg)
    elif regeneration_count > 0:
        # Subsequent regeneration attempts
        regeneration_count += 1
        msg = f"---REGENERATION ATTEMPT #{regeneration_count}---"
        print(msg)
        logger.info(msg)
    
    # Build conversation context
    conversation_context = _build_conversation_context(chat_history)
    
    # Check if the question is about roadmap
    if _is_roadmap_question(question):
        print("---DETECTED ROADMAP QUESTION---")
        course_id = _extract_course_id_from_documents(documents)
        
        if course_id:
            print(f"---FETCHING COURSE STRUCTURE FOR COURSE: {course_id}---")
            course_structures = fetch_course_structure(course_id=course_id)
            
            if course_structures:
                roadmap_text = _format_roadmap(course_structures[0])
                # Add context from documents so LLM can provide additional explanations
                context = _build_context(documents)
                enhanced_context = f"{context}\n\n---\n\n## Course Roadmap:\n\n{roadmap_text}"
                
                # Add conversation context if available - place it first
                if conversation_context:
                    enhanced_context = (
                        "INSTRUCTIONS: Use the conversation history below to understand context. "
                        "Then provide the roadmap based on the course structure.\n\n"
                        + f"{conversation_context}\n\n---\n\n{enhanced_context}"
                    )
                
                generation = generation_chain.invoke({
                    "context": enhanced_context,
                    "question": question
                })
            else:
                generation = "I couldn't find the course structure. Please make sure you're asking about a specific course."
        else:
            # No specific course_id, fetch all courses
            print("---FETCHING ALL COURSE STRUCTURES---")
            all_courses = fetch_course_structure()
            
            if all_courses:
                roadmaps = []
                for course in all_courses[:5]:  # Limit to 5 courses to avoid being too long
                    roadmaps.append(_format_roadmap(course))
                
                roadmap_text = "\n\n---\n\n".join(roadmaps)
                context = _build_context(documents)
                enhanced_context = f"{context}\n\n---\n\n## Available Courses Roadmaps:\n\n{roadmap_text}"
                
                # Add conversation context if available - place it first
                if conversation_context:
                    enhanced_context = (
                        "INSTRUCTIONS: Use the conversation history below to understand context. "
                        "Then provide the roadmaps based on the course structures.\n\n"
                        + f"{conversation_context}\n\n---\n\n{enhanced_context}"
                    )
                
                generation = generation_chain.invoke({
                    "context": enhanced_context,
                    "question": question
                })
            else:
                generation = "I couldn't find any course structures. Please try asking about a specific course."
    else:
        # Regular question
        context = _build_context(documents)
        
        # Check if this is a platform question with primarily knowledge-base documents
        # Use concise prompt for platform/knowledge-base questions
        is_platform_question = state.get("is_platform_question", False)
        kb_doc_count = sum(1 for doc in documents if (doc.metadata or {}).get("doc_type") == "knowledge_base")
        is_mostly_kb = kb_doc_count > 0 and kb_doc_count >= len(documents) * 0.5  # At least 50% KB docs
        
        if is_platform_question and is_mostly_kb:
            print("---PLATFORM QUESTION WITH KB DOCS - USING CONCISE PROMPT---")
            # For platform questions with KB docs, use concise prompt (like knowledge base docs)
            if conversation_context:
                enhanced_context = f"{conversation_context}\n\n## RETRIEVED DOCUMENTS:\n\n{context}"
            else:
                enhanced_context = context
            generation = generation_chain_platform.invoke({"context": enhanced_context, "question": question})
        else:
            # Regular generation with comprehensive prompt
            # Add conversation context if available - place it BEFORE documents for better context understanding
            if conversation_context:
                # Put conversation history first so LLM sees it first
                enhanced_context = f"{conversation_context}\n\n## RETRIEVED DOCUMENTS:\n\n{context}"
                # Add instruction to use conversation history
                enhanced_context = (
                    "INSTRUCTIONS: The user's current question may be a follow-up to previous questions. "
                    "Please read the CONVERSATION HISTORY above to understand the context. "
                    "Then answer the current question using both the conversation history and the retrieved documents below.\n\n"
                    + enhanced_context
                )
            else:
                enhanced_context = context
            
            generation = generation_chain.invoke({"context": enhanced_context, "question": question})
    
    sources = _extract_sources(documents)
    
    # Update chat history with current Q&A
    updated_history = list(chat_history) if chat_history else []
    updated_history.append((question, generation))
    
    return {
        "generation": generation,
        "documents": documents,
        "question": question,
        "sources": sources,
        "user_id": state.get("user_id"),
        "lesson_id": state.get("lesson_id"),  # Preserve lesson_id
        "is_platform_question": state.get("is_platform_question", False),  # Preserve platform question flag
        "chat_history": updated_history,
        "regeneration_count": regeneration_count,
        "web_search_count": state.get("web_search_count") or 0,  # Preserve web search count
    }
