from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Dict, List, Set, Tuple

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2 import OperationalError


def _db_config() -> Dict[str, str]:
    return {
        "host": os.getenv("RAG_DB_HOST", os.getenv("POSTGRES_HOST", "localhost")),
        "port": os.getenv("RAG_DB_PORT", os.getenv("POSTGRES_PORT", "5432")),
        "dbname": os.getenv("RAG_DB_NAME", os.getenv("POSTGRES_DB", "postgres")),
        "user": os.getenv("RAG_DB_USER", os.getenv("POSTGRES_USER", "postgres")),
        "password": os.getenv("RAG_DB_PASSWORD", os.getenv("POSTGRES_PASSWORD", "postgres")),
        "connect_timeout": int(os.getenv("RAG_DB_CONNECT_TIMEOUT", "5")),
    }


@contextmanager
def get_connection(max_retries: int = 3, retry_delay: float = 2.0):
    """
    Get a database connection with retry logic.
    
    Args:
        max_retries: Maximum number of connection retry attempts
        retry_delay: Delay in seconds between retry attempts
    """
    config = _db_config()
    last_error = None
    
    for attempt in range(max_retries):
        try:
            conn = psycopg2.connect(**config)
            try:
                yield conn
            finally:
                conn.close()
            return
        except OperationalError as e:
            last_error = e
            if attempt < max_retries - 1:
                print(f"Database connection failed (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"Database connection failed after {max_retries} attempts")
                raise
    
    # If we get here, all retries failed
    if last_error:
        raise last_error


def fetch_lessons_with_context() -> List[Dict[str, object]]:
    query = """
        SELECT
            lessons.id AS lesson_id,
            lessons.title AS lesson_title,
            lessons.content AS lesson_content,
            lessons.video_url AS lesson_video_url,
            lessons.file_url AS lesson_file_url,
            lessons.modified AS lesson_modified,
            lessons.course_id AS course_id,
            lessons.chapter_id AS chapter_id,
            chapters.title AS chapter_title,
            chapters.summary AS chapter_summary,
            chapters.modified AS chapter_modified,
            courses.title AS course_title,
            courses.short_introduction AS course_short_intro,
            courses.description AS course_description,
            courses.skill_level AS course_skill_level,
            courses.target_audience AS course_target_audience,
            courses.language AS course_language,
            courses.status AS course_status,
            courses.modified AS course_modified
        FROM lessons
        LEFT JOIN chapters ON chapters.id = lessons.chapter_id
        LEFT JOIN courses ON courses.id = lessons.course_id
        ORDER BY courses.title, chapters.position, lessons.position;
    """
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)
            return list(cur.fetchall())


def fetch_courses() -> List[Dict[str, object]]:
    query = """
        SELECT
            id AS course_id,
            title AS course_title,
            short_introduction AS course_short_intro,
            description AS course_description,
            target_audience AS course_target_audience,
            skill_level AS course_skill_level,
            language AS course_language,
            status AS course_status,
            modified AS course_modified
        FROM courses;
    """
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)
            return list(cur.fetchall())


def fetch_tags() -> Dict[Tuple[str, str], List[str]]:
    query = """
        SELECT entity_id, entity_type, array_agg(name ORDER BY name) AS names
        FROM tags
        GROUP BY entity_id, entity_type;
    """
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)
            rows = cur.fetchall()
    tags: Dict[Tuple[str, str], List[str]] = {}
    for row in rows:
        entity_id = str(row["entity_id"])
        entity_type = str(row["entity_type"])
        names = row["names"] or []
        tags[(entity_id, entity_type)] = [name for name in names if name]
    return tags


def fetch_labels() -> Dict[Tuple[str, str], List[str]]:
    query = """
        SELECT entity_id, entity_type, array_agg(name ORDER BY name) AS names
        FROM labels
        GROUP BY entity_id, entity_type;
    """
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)
            rows = cur.fetchall()
    labels: Dict[Tuple[str, str], List[str]] = {}
    for row in rows:
        entity_id = str(row["entity_id"])
        entity_type = str(row["entity_type"])
        names = row["names"] or []
        labels[(entity_id, entity_type)] = [name for name in names if name]
    return labels


def fetch_user_enrollments(user_id: str) -> Set[str]:
    query = """
        SELECT course_id
        FROM enrollments
        WHERE member_id = %s;
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (user_id,))
            rows = cur.fetchall()
    return {str(row[0]) for row in rows}


def fetch_course_structure(course_id: str | None = None) -> List[Dict[str, object]]:
    """
    Fetch course structure with chapters and lessons sorted by position.
    If course_id is provided, only fetch that course. If None, fetch all courses.
    
    Returns:
        List of dicts with structure:
        {
            "course_id": str,
            "course_title": str,
            "course_description": str,
            "course_skill_level": str,
            "chapters": [
                {
                    "chapter_id": str,
                    "chapter_title": str,
                    "chapter_summary": str,
                    "position": int,
                    "lessons": [
                        {
                            "lesson_id": str,
                            "lesson_title": str,
                            "position": int,
                            "has_video": bool,
                            "has_file": bool
                        }
                    ]
                }
            ]
        }
    """
    if course_id:
        course_filter = "WHERE courses.id = %s"
        params = (course_id,)
    else:
        course_filter = ""
        params = None
    
    query = f"""
        SELECT
            courses.id AS course_id,
            courses.title AS course_title,
            courses.description AS course_description,
            courses.skill_level AS course_skill_level,
            courses.target_audience AS course_target_audience,
            courses.language AS course_language,
            chapters.id AS chapter_id,
            chapters.title AS chapter_title,
            chapters.summary AS chapter_summary,
            chapters.position AS chapter_position,
            lessons.id AS lesson_id,
            lessons.title AS lesson_title,
            lessons.position AS lesson_position,
            CASE WHEN lessons.video_url IS NOT NULL THEN TRUE ELSE FALSE END AS has_video,
            CASE WHEN lessons.file_url IS NOT NULL THEN TRUE ELSE FALSE END AS has_file
        FROM courses
        LEFT JOIN chapters ON chapters.course_id = courses.id
        LEFT JOIN lessons ON lessons.chapter_id = chapters.id AND lessons.course_id = courses.id
        {course_filter}
        ORDER BY courses.title, chapters.position NULLS LAST, lessons.position NULLS LAST;
    """
    
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            rows = list(cur.fetchall())
    
    # Group vào structure
    courses_dict: Dict[str, Dict[str, object]] = {}
    chapters_dict: Dict[str, Dict[str, object]] = {}
    
    for row in rows:
        course_id_str = str(row["course_id"]) if row["course_id"] else None
        if not course_id_str:
            continue
            
        # Initialize course
        if course_id_str not in courses_dict:
            courses_dict[course_id_str] = {
                "course_id": course_id_str,
                "course_title": row.get("course_title"),
                "course_description": row.get("course_description"),
                "course_skill_level": row.get("course_skill_level"),
                "course_target_audience": row.get("course_target_audience"),
                "course_language": row.get("course_language"),
                "chapters": [],
            }
        
        # Add chapter if exists
        chapter_id_str = str(row["chapter_id"]) if row["chapter_id"] else None
        if chapter_id_str and chapter_id_str not in chapters_dict:
            chapter_data = {
                "chapter_id": chapter_id_str,
                "chapter_title": row.get("chapter_title"),
                "chapter_summary": row.get("chapter_summary"),
                "position": row.get("chapter_position"),
                "lessons": [],
            }
            chapters_dict[chapter_id_str] = chapter_data
            courses_dict[course_id_str]["chapters"].append(chapter_data)
        
        # Add lesson if exists
        lesson_id_str = str(row["lesson_id"]) if row["lesson_id"] else None
        if lesson_id_str and chapter_id_str:
            chapter_data = chapters_dict[chapter_id_str]
            lesson_data = {
                "lesson_id": lesson_id_str,
                "lesson_title": row.get("lesson_title"),
                "position": row.get("lesson_position"),
                "has_video": row.get("has_video", False),
                "has_file": row.get("has_file", False),
            }
            chapter_data["lessons"].append(lesson_data)
    
    return list(courses_dict.values())


def fetch_course_slug(course_id: str) -> str | None:
    """
    Fetch course slug from course_id.
    
    Args:
        course_id: The course ID to fetch slug for
        
    Returns:
        Course slug if found, None otherwise
    """
    query = """
        SELECT slug
        FROM courses
        WHERE id = %s;
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (course_id,))
            row = cur.fetchone()
            return row[0] if row else None


def fetch_courses_slugs(course_ids: List[str]) -> Dict[str, str]:
    """
    Fetch course slugs for multiple course IDs.
    
    Args:
        course_ids: List of course IDs (as strings) to fetch slugs for
        
    Returns:
        Dictionary mapping course_id (as string) to slug
    """
    if not course_ids:
        return {}
    
    query = """
        SELECT id::text, slug
        FROM courses
        WHERE id::text = ANY(%s);
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (course_ids,))
            rows = cur.fetchall()
    return {str(row[0]): row[1] for row in rows if row[1]}

