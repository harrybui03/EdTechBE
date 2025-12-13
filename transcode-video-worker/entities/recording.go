package entities

import (
	"github.com/google/uuid"
	"time"
)

type Recording struct {
	ID          uuid.UUID `json:"id"`
	LessonId    uuid.UUID `json:"lesson_id"`
	ChunkNumber int       `json:"chunk_number"`
	ChunkPath   string    `json:"chunk_path"`
	Status      string    `json:"status"`
	CreatedAt   time.Time `json:"created_at"`
	UpdatedAt   time.Time `json:"updated_at"`
}

func (Recording) TableName() string {
	return "recordings"
}

