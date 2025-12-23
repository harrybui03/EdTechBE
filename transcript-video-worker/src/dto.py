from pydantic import BaseModel


class TranscriptionMessage(BaseModel):
    jobId: str  # UUID của job
    objectPath: str  # Đường dẫn video trong MinIO (ví dụ: lessons/{lessonId}/videos/{timestamp}-{filename})
    language: str | None = None  # Optional: "vi", "zh", "ja", "en", etc. Nếu None thì auto-detect


