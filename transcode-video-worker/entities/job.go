package entities

import (
	"github.com/google/uuid"
	"time"
	"worker-transcode/constant"
)

type Job struct {
	ID         uuid.UUID          `json:"id"`
	EntityId   uuid.UUID          `json:"entity_id"`
	EntityType string             `json:"entity_type"`
	Status     constant.JobStatus `json:"status"`
	JobType    constant.JobType   `json:"job_type"`
	CreatedAt  time.Time          `json:"created_at"`
	UpdatedAt  time.Time          `json:"updated_at"`
}

func (Job) TableName() string {
	return "jobs"
}
