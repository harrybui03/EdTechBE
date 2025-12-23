import logging
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Optional


class JobRepository:
    """Repository để query job status từ PostgreSQL database"""
    
    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg["database"]
        self._logger = logging.getLogger("transcription.repository")
    
    def _get_connection(self):
        """Tạo connection đến PostgreSQL"""
        return psycopg2.connect(
            host=self._cfg["host"],
            port=self._cfg["port"],
            dbname=self._cfg["dbname"],
            user=self._cfg["user"],
            password=self._cfg["password"],
            cursor_factory=RealDictCursor
        )
    
    def find_job_by_id(self, job_id: str) -> Optional[dict]:
        """Tìm job theo ID và trả về status, entity_id"""
        try:
            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT id, entity_id, status FROM jobs WHERE id = %s",
                        (job_id,)
                    )
                    row = cur.fetchone()
                    if row:
                        return dict(row)
                    return None
            finally:
                conn.close()
        except Exception as e:
            self._logger.error(f"❌ Failed to query job {job_id[:8]}...: {e}")
            raise

