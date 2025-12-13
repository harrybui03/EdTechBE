package entities

import "github.com/google/uuid"

type Lesson struct {
	Id       uuid.UUID `json:"id"`
	VideoUrl string    `json:"video_url"`
}

func (Lesson) TableName() string {
	return "lessons"
}
