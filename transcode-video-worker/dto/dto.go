package dto

import "github.com/google/uuid"

type JobMessage struct {
	JobId      uuid.UUID `json:"jobId"`
	ObjectPath string    `json:"objectPath"`
	FileName   string    `json:"fileName"`
}

type RecordingMergeMessage struct {
	JobId         uuid.UUID `json:"jobId"`
	LiveSessionId uuid.UUID `json:"liveSessionId"`
}