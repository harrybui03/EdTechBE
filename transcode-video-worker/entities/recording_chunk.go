package entities

import (
	"github.com/google/uuid"
	"time"
)

type RecordingChunk struct {
	ID             uuid.UUID `json:"id" gorm:"type:uuid;primary_key;default:gen_random_uuid()"`
	LiveSessionId uuid.UUID `json:"live_session_id" gorm:"type:uuid;not null;index:idx_recording_chunks_session"`
	ChunkIndex     int       `json:"chunk_index" gorm:"not null"`
	ObjectName     string    `json:"object_name" gorm:"type:varchar(500);not null"`
	FileSize       *int64    `json:"file_size" gorm:"type:bigint"`
	DurationSeconds *int     `json:"duration_seconds" gorm:"type:integer"`
	Status         string    `json:"status" gorm:"type:varchar(20);not null;check:status IN ('UPLOADED', 'PROCESSING', 'COMPLETED', 'FAILED')"`
	CreatedAt      time.Time `json:"created_at" gorm:"type:timestamptz;not null;default:CURRENT_TIMESTAMP"`
}

func (RecordingChunk) TableName() string {
	return "recording_chunks"
}

