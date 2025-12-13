from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.documents import Document

from database import (
    fetch_courses,
    fetch_labels,
    fetch_lessons_with_context,
    fetch_tags,
)

# Only load .env file if not in Docker (override=False prevents overriding existing env vars)
# In Docker, environment variables are set by docker-compose.yml
load_dotenv(override=False)

CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION_NAME", "rag-edtech")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./.chroma")
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "120"))
TRANSCRIPTS_DIR = os.getenv("TRANSCRIPTS_DIR", "../../transcription-worker/transcripts")

# Resolve knowledge-base directory relative to project root (not current working directory)
# Default: ../knowledge-base (from agentic_rag/ directory) or ./knowledge-base (from project root)
_default_kb_dir = os.getenv("KNOWLEDGE_BASE_DIR")
if _default_kb_dir is None:
    # Try to find knowledge-base relative to this file's location
    _current_file_dir = Path(__file__).parent  # agentic_rag/
    _project_root = _current_file_dir.parent  # project root
    _kb_path = _project_root / "knowledge-base"
    if _kb_path.exists():
        _default_kb_dir = str(_kb_path)
    else:
        _default_kb_dir = "./knowledge-base"  # Fallback to relative path

KNOWLEDGE_BASE_DIR = os.getenv("KNOWLEDGE_BASE_DIR", _default_kb_dir)


def _isoformat(value: Optional[object]) -> Optional[str]:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _combine_taxonomy(
    entity_id: Optional[str],
    entity_type: str,
    tags: Dict[Tuple[str, str], List[str]],
    labels: Dict[Tuple[str, str], List[str]],
) -> List[str]:
    if not entity_id:
        return []
    key = (entity_id, entity_type)
    combined = set(tags.get(key, [])) | set(labels.get(key, []))
    return sorted(tag for tag in combined if tag)


def _sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        elif isinstance(value, (list, tuple, set)):
            flattened = [str(item) for item in value if item not in (None, "")]
            if flattened:
                sanitized[key] = "; ".join(flattened)
        else:
            sanitized[key] = str(value)
    return sanitized


def _build_course_documents(
    courses: List[Dict[str, object]],
    tags: Dict[Tuple[str, str], List[str]],
    labels: Dict[Tuple[str, str], List[str]],
) -> List[Document]:
    documents: List[Document] = []
    for course in courses:
        course_id = str(course["course_id"])
        content_parts = [
            f"Course: {course.get('course_title')}",
            course.get("course_short_intro") or "",
            course.get("course_description") or "",
            f"Target audience: {course.get('course_target_audience') or 'N/A'}",
            f"Skill level: {course.get('course_skill_level') or 'N/A'}",
        ]
        content = "\n\n".join(part for part in content_parts if part).strip()
        if not content:
            continue

        taxonomy = _combine_taxonomy(course_id, "Course", tags, labels)
        metadata = {
            "document_id": f"course:{course_id}",
            "doc_type": "course_overview",
            "course_id": course_id,
            "course_title": course.get("course_title"),
            "language": course.get("course_language"),
            "requires_enrollment": False,
            "tags": taxonomy,
            "last_modified": _isoformat(course.get("course_modified")),
            "course_status": course.get("course_status"),
        }
        documents.append(
            Document(page_content=content, metadata=_sanitize_metadata(metadata))
        )
    return documents


def _build_lesson_documents(
    lessons: List[Dict[str, object]],
    tags: Dict[Tuple[str, str], List[str]],
    labels: Dict[Tuple[str, str], List[str]],
) -> List[Document]:
    documents: List[Document] = []
    for lesson in lessons:
        lesson_id = lesson.get("lesson_id")
        course_id = lesson.get("course_id")
        chapter_id = lesson.get("chapter_id")

        lesson_content = lesson.get("lesson_content") or ""
        chapter_summary = lesson.get("chapter_summary") or ""

        content_parts = [
            f"Course: {lesson.get('course_title')}",
            f"Chapter: {lesson.get('chapter_title') or 'N/A'}",
            f"Lesson: {lesson.get('lesson_title')}",
            chapter_summary,
            lesson_content,
        ]

        if lesson.get("lesson_video_url"):
            content_parts.append(f"Video URL: {lesson['lesson_video_url']}")
        if lesson.get("lesson_file_url"):
            content_parts.append(f"Attachment URL: {lesson['lesson_file_url']}")

        content = "\n\n".join(part for part in content_parts if part).strip()
        if not content:
            continue

        lesson_id_str = str(lesson_id) if lesson_id else None
        course_id_str = str(course_id) if course_id else None
        chapter_id_str = str(chapter_id) if chapter_id else None

        taxonomy = (
            _combine_taxonomy(lesson_id_str, "Lesson", tags, labels)
            + _combine_taxonomy(chapter_id_str, "Chapter", tags, labels)
            + _combine_taxonomy(course_id_str, "Course", tags, labels)
        )

        metadata = {
            "document_id": f"lesson:{lesson_id_str}",
            "doc_type": "lesson",
            "course_id": course_id_str,
            "course_title": lesson.get("course_title"),
            "chapter_id": chapter_id_str,
            "chapter_title": lesson.get("chapter_title"),
            "lesson_id": lesson_id_str,
            "lesson_title": lesson.get("lesson_title"),
            "requires_enrollment": True,
            "tags": sorted(set(taxonomy)),
            "last_modified": _isoformat(lesson.get("lesson_modified")),
            "chapter_summary": chapter_summary,
            "course_skill_level": lesson.get("course_skill_level"),
            "course_language": lesson.get("course_language"),
        }
        documents.append(
            Document(page_content=content, metadata=_sanitize_metadata(metadata))
        )
    return documents


def _load_transcript_files() -> List[Dict[str, Any]]:
    """Load all transcript files from the transcripts directory"""
    transcripts_dir = Path(TRANSCRIPTS_DIR)
    # Resolve to absolute path for better error messages
    transcripts_dir = transcripts_dir.resolve()
    
    if not transcripts_dir.exists():
        print(f"[INGEST] Transcripts directory not found: {transcripts_dir}")
        print(f"[INGEST] Please check TRANSCRIPTS_DIR environment variable or ensure the directory exists")
        return []
    
    transcripts = []
    for transcript_file in transcripts_dir.glob("*.json"):
        try:
            with open(transcript_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                lesson_id = data.get("lessonId", "unknown")
                text_preview = (data.get("translatedText") or data.get("text", ""))[:100]
                duration = data.get("duration", 0)
                language = data.get("language", "unknown")
                segments_count = len(data.get("segments", []))
                print(
                    f"[INGEST] Loaded transcript | "
                    f"file={transcript_file.name} | "
                    f"lessonId={lesson_id[:8] if len(str(lesson_id)) > 8 else lesson_id}... | "
                    f"language={language} | "
                    f"duration={duration:.1f}s | "
                    f"segments={segments_count} | "
                    f"preview={text_preview}..."
                )
                transcripts.append(data)
        except Exception as e:
            print(f"[INGEST] Failed to load transcript {transcript_file}: {e}")
            continue
    
    print(f"[INGEST] Total transcripts loaded: {len(transcripts)}")
    return transcripts


def _build_transcript_documents(
    transcripts: List[Dict[str, Any]],
    lessons: List[Dict[str, object]],
    tags: Dict[Tuple[str, str], List[str]],
    labels: Dict[Tuple[str, str], List[str]],
) -> List[Document]:
    """Tạo Document objects từ transcript data"""
    documents: List[Document] = []
    
    # Tạo map từ lesson_id -> lesson metadata để lookup nhanh
    lesson_map: Dict[str, Dict[str, object]] = {}
    for lesson in lessons:
        lesson_id_str = str(lesson.get("lesson_id", ""))
        if lesson_id_str:
            lesson_map[lesson_id_str] = lesson
    
    for transcript in transcripts:
        lesson_id = transcript.get("lessonId")
        if not lesson_id:
            print(f"[INGEST] Skipping transcript: missing lessonId")
            continue
        
        lesson_id_str = str(lesson_id)
        lesson_meta = lesson_map.get(lesson_id_str)
        
        if not lesson_meta:
            print(
                f"[INGEST] Warning: Transcript lessonId={lesson_id_str[:8]}... "
                f"not found in database, will use transcript data only"
            )
        
        # Lấy transcript text (ưu tiên translatedText nếu có, nếu không thì dùng text gốc)
        transcript_text = transcript.get("translatedText") or transcript.get("text", "")
        if not transcript_text:
            print(f"[INGEST] Skipping transcript lessonId={lesson_id_str[:8]}...: empty text")
            continue
        
        # Lấy segments để có thể format tốt hơn
        segments = transcript.get("segments", [])
        translated_segments = transcript.get("translatedSegments", [])
        
        # Build content từ transcript
        content_parts = []
        
        # Thêm context từ lesson nếu có
        if lesson_meta:
            content_parts.append(f"Course: {lesson_meta.get('course_title', 'N/A')}")
            content_parts.append(f"Chapter: {lesson_meta.get('chapter_title', 'N/A')}")
            content_parts.append(f"Lesson: {lesson_meta.get('lesson_title', 'N/A')}")
            if lesson_meta.get("chapter_summary"):
                content_parts.append(f"Chapter Summary: {lesson_meta['chapter_summary']}")
        
        # Thêm transcript text
        content_parts.append("Audio Transcript:")
        content_parts.append(transcript_text)
        
        # Thêm segments nếu có (để có timing info)
        if translated_segments:
            segments_text = "\n".join(
                f"[{seg.get('start', 0):.1f}s - {seg.get('end', 0):.1f}s] {seg.get('text', '')}"
                for seg in translated_segments
            )
            if segments_text:
                content_parts.append("\nTranscript Segments:")
                content_parts.append(segments_text)
        elif segments:
            segments_text = "\n".join(
                f"[{seg.get('start', 0):.1f}s - {seg.get('end', 0):.1f}s] {seg.get('text', '')}"
                for seg in segments
            )
            if segments_text:
                content_parts.append("\nTranscript Segments:")
                content_parts.append(segments_text)
        
        content = "\n\n".join(part for part in content_parts if part).strip()
        if not content:
            continue
        
        # Build metadata
        course_id_str = str(lesson_meta.get("course_id", "")) if lesson_meta else None
        chapter_id_str = str(lesson_meta.get("chapter_id", "")) if lesson_meta else None
        
        taxonomy = (
            _combine_taxonomy(lesson_id_str, "Lesson", tags, labels)
            + _combine_taxonomy(chapter_id_str, "Chapter", tags, labels)
            + _combine_taxonomy(course_id_str, "Course", tags, labels)
        )
        
        metadata = {
            "document_id": f"transcript:{lesson_id_str}",
            "doc_type": "transcript",
            "course_id": course_id_str,
            "course_title": lesson_meta.get("course_title") if lesson_meta else None,
            "chapter_id": chapter_id_str,
            "chapter_title": lesson_meta.get("chapter_title") if lesson_meta else None,
            "lesson_id": lesson_id_str,
            "lesson_title": lesson_meta.get("lesson_title") if lesson_meta else None,
            "requires_enrollment": True,
            "tags": sorted(set(taxonomy)),
            "transcript_language": transcript.get("language"),
            "transcript_model": transcript.get("model", "assemblyai"),
            "transcript_duration": transcript.get("duration"),
            "transcript_created_at": transcript.get("createdAt"),
            "has_translation": bool(transcript.get("translatedText")),
            "course_skill_level": lesson_meta.get("course_skill_level") if lesson_meta else None,
            "course_language": lesson_meta.get("course_language") if lesson_meta else None,
        }
        
        documents.append(
            Document(page_content=content, metadata=_sanitize_metadata(metadata))
        )
        
        # Log transcript document được tạo
        lesson_title = lesson_meta.get("lesson_title") if lesson_meta else "N/A"
        course_title = lesson_meta.get("course_title") if lesson_meta else "N/A"
        text_length = len(transcript_text)
        print(
            f"[INGEST] Created transcript document | "
            f"lessonId={lesson_id_str[:8]}... | "
            f"course={course_title[:30] if course_title else 'N/A'} | "
            f"lesson={lesson_title[:30] if lesson_title else 'N/A'} | "
            f"textLength={text_length} chars | "
            f"segments={len(translated_segments) if translated_segments else len(segments)} | "
            f"hasTranslation={bool(transcript.get('translatedText'))}"
        )
    
    print(f"[INGEST] Total transcript documents created: {len(documents)}")
    return documents


def _load_markdown_files() -> List[Dict[str, Any]]:
    """Load all markdown files from the knowledge-base directory"""
    kb_dir = Path(KNOWLEDGE_BASE_DIR)
    # Resolve to absolute path for better error messages
    kb_dir = kb_dir.resolve()
    
    if not kb_dir.exists():
        print(f"[INGEST] Knowledge base directory not found: {kb_dir}")
        print(f"[INGEST] Please check KNOWLEDGE_BASE_DIR environment variable or ensure the directory exists")
        return []
    
    markdown_files = []
    # Recursively find all .md files
    for md_file in kb_dir.rglob("*.md"):
        # Skip README.md files (they're documentation, not guides)
        if md_file.name.lower() == "readme.md":
            continue
            
        try:
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()
                
            # Extract relative path from knowledge-base root
            try:
                relative_path = md_file.relative_to(kb_dir)
            except ValueError:
                # If can't get relative path, use filename
                relative_path = Path(md_file.name)
            
            # Determine category from directory structure
            category = "unknown"
            parts = relative_path.parts
            if len(parts) > 0:
                parent_dir = parts[0].lower()
                if "instructor" in parent_dir or "instructor-guide" in parent_dir:
                    category = "instructor_guide"
                elif "user" in parent_dir or "user-guide" in parent_dir:
                    category = "user_guide"
                elif "faq" in parent_dir:
                    category = "faq"
            
            # Extract title from first H1 heading
            title = None
            lines = content.split("\n")
            for line in lines:
                line_stripped = line.strip()
                if line_stripped.startswith("# ") and len(line_stripped) > 2:
                    title = line_stripped[2:].strip()
                    break
            
            # If no H1 found, use filename (without extension) as title
            if not title:
                title = md_file.stem.replace("-", " ").replace("_", " ").title()
            
            markdown_files.append({
                "file_path": str(relative_path),
                "absolute_path": str(md_file),
                "content": content,
                "title": title,
                "category": category,
                "last_modified": md_file.stat().st_mtime,
            })
            
            print(
                f"[INGEST] Loaded markdown | "
                f"file={relative_path} | "
                f"category={category} | "
                f"title={title[:50] if len(title) > 50 else title} | "
                f"size={len(content)} chars"
            )
        except Exception as e:
            print(f"[INGEST] Failed to load markdown {md_file}: {e}")
            continue
    
    print(f"[INGEST] Total markdown files loaded: {len(markdown_files)}")
    return markdown_files


def _build_knowledge_documents(
    markdown_files: List[Dict[str, Any]]
) -> List[Document]:
    """Create Document objects from markdown knowledge base files"""
    documents: List[Document] = []
    
    for md_data in markdown_files:
        content = md_data["content"]
        if not content or not content.strip():
            continue
        
        # Use the full markdown content
        # The text splitter will handle chunking later
        page_content = content.strip()
        
        metadata = {
            "document_id": f"knowledge_base:{md_data['file_path']}",
            "doc_type": "knowledge_base",
            "category": md_data["category"],
            "file_path": md_data["file_path"],
            "title": md_data["title"],
            "last_modified": _isoformat(md_data.get("last_modified")),
            "requires_enrollment": False,  # Knowledge base is public
        }
        
        documents.append(
            Document(
                page_content=page_content,
                metadata=_sanitize_metadata(metadata)
            )
        )
        
        print(
            f"[INGEST] Created knowledge base document | "
            f"file={md_data['file_path']} | "
            f"category={md_data['category']} | "
            f"title={md_data['title'][:50] if len(md_data['title']) > 50 else md_data['title']} | "
            f"contentLength={len(page_content)} chars"
        )
    
    print(f"[INGEST] Total knowledge base documents created: {len(documents)}")
    return documents


def load_documents() -> List[Document]:
    lessons = fetch_lessons_with_context()
    courses = fetch_courses()
    tags = fetch_tags()
    labels = fetch_labels()
    
    # Load transcripts từ thư mục transcripts
    transcripts = _load_transcript_files()
    
    # Load markdown knowledge base files
    markdown_files = _load_markdown_files()

    documents = []
    documents.extend(_build_course_documents(courses, tags, labels))
    documents.extend(_build_lesson_documents(lessons, tags, labels))
    documents.extend(_build_transcript_documents(transcripts, lessons, tags, labels))
    documents.extend(_build_knowledge_documents(markdown_files))
    return documents


def build_vectorstore() -> Chroma:
    raw_documents = load_documents()
    if not raw_documents:
        raise RuntimeError("No documents fetched from the database for ingestion.")

    print(f"[INGEST] Raw documents fetched: {len(raw_documents)}")
    
    # Đếm theo doc_type
    doc_counts = {}
    for doc in raw_documents:
        doc_type = doc.metadata.get("doc_type", "unknown")
        doc_counts[doc_type] = doc_counts.get(doc_type, 0) + 1
    print(f"[INGEST] Document breakdown: {doc_counts}")
    
    # Preview một số documents
    for idx, doc in enumerate(raw_documents[:5], start=1):
        preview = (doc.page_content or "").strip()
        if len(preview) > 200:
            preview = preview[:200].rstrip() + "..."
        
        doc_info = {
            "document_id": doc.metadata.get("document_id"),
            "doc_type": doc.metadata.get("doc_type"),
            "course_id": doc.metadata.get("course_id"),
            "lesson_id": doc.metadata.get("lesson_id"),
            "title": doc.metadata.get("lesson_title") or doc.metadata.get("course_title"),
            "preview": preview,
        }
        
        # Thêm thông tin đặc biệt cho transcript
        if doc.metadata.get("doc_type") == "transcript":
            doc_info["transcript_language"] = doc.metadata.get("transcript_language")
            doc_info["transcript_duration"] = doc.metadata.get("transcript_duration")
            doc_info["has_translation"] = doc.metadata.get("has_translation")
        
        print(f"[INGEST] Preview {idx}: {doc_info}")

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    doc_splits = text_splitter.split_documents(raw_documents)
    print(f"[INGEST] Chunks generated: {len(doc_splits)}")

    embedding = FastEmbedEmbeddings()
    # Reset collection before re-ingesting
    try:
        Chroma(
            collection_name=CHROMA_COLLECTION,
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embedding,
        ).delete_collection()
    except ValueError:
        # Collection may not exist yet on first run
        pass

    vector_store = Chroma.from_documents(
        documents=doc_splits,
        collection_name=CHROMA_COLLECTION,
        embedding=embedding,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    return vector_store


vectorstore = build_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 7})

# Note: langchain_chroma does not support 'where' filter in search_kwargs
# Metadata filtering is done in retrieve.py node using post-filter approach
# This is still efficient as it filters after retrieval but before batch grading
