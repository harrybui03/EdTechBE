"""
FastAPI server for Agentic RAG API.
Provides REST API endpoints for the frontend to interact with the RAG system.
"""

import io
import logging
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict

# Ensure we can import graph module
# When running as a package (agentic_rag.api), use relative import
# When running directly, add path to sys.path
try:
    from .graph.graph import app
    from .database import fetch_courses_slugs
except ImportError:
    # Fallback: add agentic_rag to path if running directly
    current_file = Path(__file__).resolve()
    agentic_rag_dir = current_file.parent
    if str(agentic_rag_dir) not in sys.path:
        sys.path.insert(0, str(agentic_rag_dir))
    from graph.graph import app
    from database import fetch_courses_slugs

# Only load .env file if not in Docker (override=False prevents overriding existing env vars)
# In Docker, environment variables are set by docker-compose.yml
load_dotenv(override=False)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("agentic_rag.api")

# Initialize FastAPI app
api_app = FastAPI(
    title="Agentic RAG API",
    description="API for querying the Agentic RAG system",
    version="1.0.0",
)

# Configure CORS
api_app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://localhost:3000",  # Alternative port
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class ChatMessage(BaseModel):
    """Single chat message in conversation history"""

    question: Optional[str] = Field(None, description="User's question")
    answer: Optional[str] = Field(None, description="Assistant's answer")


class AskRequest(BaseModel):
    """Request model for asking a question"""
    
    model_config = ConfigDict(populate_by_name=True)  # Allow both snake_case and camelCase

    question: str = Field(..., description="The question to ask", min_length=1)
    user_id: Optional[str] = Field(None, description="Optional user ID for permission checks")
    lesson_id: Optional[str] = Field(None, alias="lessonId", description="Optional lesson ID to filter documents to specific lesson")
    chat_history: Optional[List[ChatMessage]] = Field(
        None, description="Previous conversation history"
    )


class Source(BaseModel):
    """Source document metadata"""

    rank: Optional[int] = Field(None, description="Rank of the document in retrieval results")
    document_id: Optional[str] = None
    doc_type: Optional[str] = None
    course_id: Optional[str] = None
    course_title: Optional[str] = None
    chapter_id: Optional[str] = None
    chapter_title: Optional[str] = None
    lesson_id: Optional[str] = None
    lesson_title: Optional[str] = None
    requires_enrollment: Optional[bool] = None
    tags: Optional[List[str]] = None
    language: Optional[str] = None
    course_skill_level: Optional[str] = None
    chapter_summary: Optional[str] = None
    last_modified: Optional[str] = None
    distance: Optional[float] = Field(None, description="Similarity distance score")
    metadata: Optional[dict] = Field(None, description="Additional metadata")


class AskResponse(BaseModel):
    """Response model for ask endpoint"""

    answer: str = Field(..., description="The generated answer")
    trace: str = Field(..., description="Debug trace output from the RAG pipeline")
    sources: List[Source] = Field(default_factory=list, description="Source documents metadata")
    chat_history: List[ChatMessage] = Field(
        default_factory=list, description="Updated conversation history"
    )


@api_app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Agentic RAG API is running", "version": "1.0.0"}


@api_app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}


@api_app.post("/api/v1/rag/ask", response_model=AskResponse)
async def ask_question(request: AskRequest) -> AskResponse:
    """
    Ask a question to the Agentic RAG system.
    
    This endpoint processes the question through the RAG pipeline and returns:
    - The generated answer
    - Source documents metadata
    - Updated conversation history
    - Debug trace information
    
    Args:
        request: AskRequest containing question, optional user_id and chat_history
        
    Returns:
        AskResponse with answer, sources, chat_history, and trace
        
    Raises:
        HTTPException: If question is empty or processing fails
    """
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        logger.info(f"[ASK] Received question: {request.question[:100]}...")
        logger.info(f"[ASK] User ID: {request.user_id}, Lesson ID: {request.lesson_id}")
        
        # Prepare payload for the graph
        payload = {"question": request.question.strip()}
        
        if request.user_id and request.user_id.strip():
            payload["user_id"] = request.user_id.strip()
        
        if request.lesson_id and request.lesson_id.strip():
            payload["lesson_id"] = request.lesson_id.strip()
        
        # Convert chat_history from Pydantic models to tuples
        if request.chat_history:
            payload["chat_history"] = [
                (msg.question, msg.answer) for msg in request.chat_history
            ]
            logger.info(f"[ASK] Chat history: {len(request.chat_history)} messages")

        # Capture stdout for trace output
        buf = io.StringIO()
        with redirect_stdout(buf):
            # Increase recursion limit to handle complex flows
            # Also add config to prevent infinite loops
            logger.info("[ASK] Invoking RAG graph...")
            result = app.invoke(
                input=payload,
                config={"recursion_limit": 30}  # Increased from default 25
            )
            logger.info("[ASK] RAG graph execution completed")

        trace = buf.getvalue()
        logger.info(f"[ASK] Trace captured: {len(trace)} characters")
        answer = result.get("generation", str(result))
        sources_raw = result.get("sources", [])
        updated_history_raw = result.get("chat_history", [])

        # Convert sources to Pydantic models
        sources = []
        for source_raw in sources_raw:
            if isinstance(source_raw, dict):
                sources.append(Source(**source_raw))
            else:
                # Fallback for string sources
                sources.append(Source(metadata={"raw": str(source_raw)}))

        # Convert chat_history from tuples to Pydantic models
        chat_history = []
        for q, a in updated_history_raw:
            chat_history.append(ChatMessage(question=q, answer=a))

        logger.info(f"[ASK] Response prepared: answer length={len(answer)}, sources={len(sources)}")
        return AskResponse(
            answer=answer,
            trace=trace,
            sources=sources,
            chat_history=chat_history,
        )

    except Exception as e:
        logger.error(f"[ASK] Error processing question: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}",
        )


# New endpoint models for /api/v1/ask
class AskV1Request(BaseModel):
    """Request model for /api/v1/ask endpoint"""
    
    model_config = ConfigDict(populate_by_name=True)  # Allow both snake_case and camelCase

    question: str = Field(..., description="The question to ask", min_length=1)
    user_id: str = Field(..., alias="userId", description="User ID")
    lesson_id: Optional[str] = Field(None, alias="lessonId", description="Optional lesson ID to filter documents to specific lesson")
    chat_history: Optional[List[ChatMessage]] = Field(
        default=None, alias="chatHistory", description="Previous conversation history (only last 5 will be used)"
    )


class CourseSource(BaseModel):
    """Source course metadata for /api/v1/ask response"""

    title: str = Field(..., description="Course title")
    slug: str = Field(..., description="Course slug (link to course)")


class LessonSource(BaseModel):
    """
    Source lesson metadata for /api/v1/ask response when lesson_id is provided.
    
    NOTE: The primary content of a lesson is the video transcript.
    This model prioritizes transcript documents over lesson metadata documents.
    """

    lesson_id: str = Field(..., description="Lesson ID")
    lesson_title: Optional[str] = Field(None, description="Lesson title")
    course_id: Optional[str] = Field(None, description="Course ID")
    course_title: Optional[str] = Field(None, description="Course title")
    course_slug: Optional[str] = Field(None, description="Course slug (link to course)")
    doc_type: str = Field(..., description="Document type: 'transcript' (primary content) or 'lesson' (metadata fallback if no transcript)")


class AskV1Response(BaseModel):
    """Response model for /api/v1/ask endpoint"""

    answer: str = Field(..., description="The generated answer")
    sources: List[Union[CourseSource, LessonSource]] = Field(
        default_factory=list, 
        description="Source courses (when no lesson_id) or lesson sources (when lesson_id is provided)"
    )


@api_app.post("/api/v1/ask", response_model=AskV1Response)
async def ask_v1(request: AskV1Request) -> AskV1Response:
    """
    Ask a question to the Agentic RAG system (v1 endpoint).
    
    This endpoint processes the question through the RAG pipeline and returns:
    - The generated answer
    - Source courses (when no lesson_id) or lesson sources (when lesson_id is provided)
    
    Only the last 5 questions from chat_history will be used.
    
    Source filtering logic:
    - If lesson_id is provided: Returns sources from that specific lesson (prioritizes "transcript" as main content, falls back to "lesson" if no transcript)
    - If lesson_id is not provided: Returns course sources (doc_type == "course_overview")
    
    Args:
        request: AskV1Request containing question, user_id, optional lesson_id and chat_history
        
    Returns:
        AskV1Response with answer and sources (courses or lessons depending on lesson_id)
        
    Raises:
        HTTPException: If question is empty or processing fails
    """
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        logger.info(f"[ASK_V1] Received question: {request.question[:100]}...")
        logger.info(f"[ASK_V1] User ID: {request.user_id}, Lesson ID: {request.lesson_id}")
        
        # Prepare payload for the graph
        payload = {"question": request.question.strip()}
        
        if request.user_id and request.user_id.strip():
            payload["user_id"] = request.user_id.strip()
        
        if request.lesson_id and request.lesson_id.strip():
            payload["lesson_id"] = request.lesson_id.strip()
        
        # Only take last 5 questions from chat_history
        # Filter out messages with null question or answer
        if request.chat_history:
            last_5_history = request.chat_history[-5:]
            payload["chat_history"] = [
                (msg.question, msg.answer) 
                for msg in last_5_history 
                if msg.question and msg.answer  # Skip null/empty messages
            ]
            logger.info(f"[ASK_V1] Chat history: {len(payload.get('chat_history', []))} messages (last 5)")

        # Capture stdout for trace output (but don't include in response)
        # Trace will be logged to console via print statements in graph nodes
        buf = io.StringIO()
        with redirect_stdout(buf):
            logger.info("[ASK_V1] Invoking RAG graph...")
            result = app.invoke(
                input=payload,
                config={"recursion_limit": 30}
            )
            logger.info("[ASK_V1] RAG graph execution completed")
        
        # Log trace to console for debugging (even though not in response)
        trace = buf.getvalue()
        if trace:
            logger.info(f"[ASK_V1] Agentic trace:\n{trace}")

        answer = result.get("generation", str(result))
        sources_raw = result.get("sources", [])
        lesson_id = request.lesson_id.strip() if request.lesson_id and request.lesson_id.strip() else None

        sources = []
        
        if lesson_id:
            # When lesson_id is provided: Return lesson sources
            # NOTE: Transcript is the PRIMARY content of a lesson (video transcript)
            # We prioritize transcript over lesson document metadata
            lesson_sources_map: Dict[str, Dict[str, any]] = {}
            
            # Collect lesson documents - prioritize transcript (main content) over lesson metadata
            for source_raw in sources_raw:
                if isinstance(source_raw, dict):
                    doc_type = source_raw.get("doc_type")
                    doc_lesson_id = source_raw.get("lesson_id")
                    
                    # Only include documents from the specified lesson
                    if doc_lesson_id == lesson_id and doc_type in ["lesson", "transcript"]:
                        lesson_id_str = str(doc_lesson_id)
                        course_id = source_raw.get("course_id")
                        course_title = source_raw.get("course_title")
                        lesson_title = source_raw.get("lesson_title")
                        
                        # Priority: transcript (main content) > lesson (metadata fallback)
                        if lesson_id_str not in lesson_sources_map:
                            # First document found - store it
                            lesson_sources_map[lesson_id_str] = {
                                "lesson_id": lesson_id_str,
                                "lesson_title": lesson_title,
                                "course_id": str(course_id) if course_id else None,
                                "course_title": course_title,
                                "doc_type": doc_type
                            }
                        elif doc_type == "transcript" and lesson_sources_map[lesson_id_str]["doc_type"] == "lesson":
                            # Replace lesson metadata with transcript (transcript is main content)
                            lesson_sources_map[lesson_id_str] = {
                                "lesson_id": lesson_id_str,
                                "lesson_title": lesson_title,
                                "course_id": str(course_id) if course_id else None,
                                "course_title": course_title,
                                "doc_type": "transcript"
                            }
            
            # Fetch course slugs for lesson sources
            course_ids_for_slugs = [
                info["course_id"] 
                for info in lesson_sources_map.values() 
                if info.get("course_id")
            ]
            if course_ids_for_slugs:
                course_slugs = fetch_courses_slugs(course_ids_for_slugs)
                for lesson_id_str, lesson_info in lesson_sources_map.items():
                    course_id_str = lesson_info.get("course_id")
                    if course_id_str:
                        lesson_info["course_slug"] = course_slugs.get(course_id_str)
            
            # Build sources list with LessonSource
            for lesson_id_str, lesson_info in lesson_sources_map.items():
                sources.append(LessonSource(
                    lesson_id=lesson_info["lesson_id"],
                    lesson_title=lesson_info.get("lesson_title"),
                    course_id=lesson_info.get("course_id"),
                    course_title=lesson_info.get("course_title"),
                    course_slug=lesson_info.get("course_slug"),
                    doc_type=lesson_info["doc_type"]
                ))
            
            logger.info(f"[ASK_V1] Response prepared: answer length={len(answer)}, lesson sources={len(sources)} (lesson_id={lesson_id})")
        else:
            # When no lesson_id: Return course sources (course_overview documents)
            course_ids: Set[str] = set()
            course_sources_map: Dict[str, Dict[str, str]] = {}
            
            for source_raw in sources_raw:
                if isinstance(source_raw, dict):
                    doc_type = source_raw.get("doc_type")
                    # Only include course_overview documents
                    if doc_type == "course_overview":
                        course_id = source_raw.get("course_id")
                        course_title = source_raw.get("course_title")
                        
                        if course_id and course_title:
                            course_id_str = str(course_id)
                            course_ids.add(course_id_str)
                            # Store course info, will update with slug later
                            if course_id_str not in course_sources_map:
                                course_sources_map[course_id_str] = {
                                    "title": course_title
                                }
            
            # Fetch slugs for all course_ids
            if course_ids:
                course_slugs = fetch_courses_slugs(list(course_ids))
                
                # Build sources list with title and slug
                for course_id_str, course_info in course_sources_map.items():
                    slug = course_slugs.get(course_id_str)
                    if slug:  # Only include if slug exists
                        sources.append(CourseSource(
                            title=course_info["title"],
                            slug=slug
                        ))
            
            logger.info(f"[ASK_V1] Response prepared: answer length={len(answer)}, course sources={len(sources)}")
        return AskV1Response(
            answer=answer,
            sources=sources,
        )

    except Exception as e:
        logger.error(f"[ASK_V1] Error processing question: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "agentic_rag.api:api_app",
        host="0.0.0.0",
        port=8002,  # Port 8002 to avoid conflict with Spring Boot backend (port 8000)
        reload=True,
    )

