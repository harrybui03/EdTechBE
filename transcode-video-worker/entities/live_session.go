package entities

import (
	"github.com/google/uuid"
	"time"
)

type LiveSession struct {
	ID                    uuid.UUID  `json:"id" gorm:"type:uuid;primary_key;default:gen_random_uuid()"`
	JanusSessionId        *int64     `json:"janus_session_id" gorm:"type:bigint"`
	JanusHandleId         *int64     `json:"janus_handle_id" gorm:"type:bigint"`
	RoomId                int64      `json:"room_id" gorm:"type:bigint;not null;uniqueIndex:unique_room_id"`
	InstructorId          uuid.UUID  `json:"instructor_id" gorm:"type:uuid;not null;index:idx_live_sessions_instructor_id"`
	BatchId               *uuid.UUID `json:"batch_id" gorm:"type:uuid;index:idx_live_sessions_batch_id"`
	Status                string     `json:"status" gorm:"type:varchar(20);not null;default:'PUBLISHED';index:idx_live_sessions_status"`
	Title                 *string    `json:"title" gorm:"type:varchar(255)"`
	Description           *string    `json:"description" gorm:"type:text"`
	StartedAt             *time.Time `json:"started_at" gorm:"type:timestamptz"`
	EndedAt               *time.Time `json:"ended_at" gorm:"type:timestamptz"`
	CreatedAt             time.Time `json:"created_at" gorm:"type:timestamptz;not null;default:CURRENT_TIMESTAMP"`
	UpdatedAt             time.Time `json:"updated_at" gorm:"type:timestamptz;not null;default:CURRENT_TIMESTAMP"`
	
	// Recording fields
	RecordingStatus       string     `json:"recording_status" gorm:"type:varchar(20);default:'NOT_STARTED'"`
	FinalVideoObjectName  *string   `json:"final_video_object_name" gorm:"type:varchar(500)"`
	RecordingDuration     *int      `json:"recording_duration" gorm:"type:integer"`
	TotalChunks           int       `json:"total_chunks" gorm:"type:integer;default:0"`
}

func (LiveSession) TableName() string {
	return "live_sessions"
}

