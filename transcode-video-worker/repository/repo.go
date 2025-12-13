package repository

import (
	"context"
	"database/sql"
	"github.com/google/uuid"
	"gorm.io/driver/postgres"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"
	"worker-transcode/constant"
	"worker-transcode/entities"
)

type JobRepository interface {
	Transaction(ctx context.Context, callback func(ctx context.Context) error, opts ...*sql.TxOptions) error
	GetDB() *gorm.DB
	FindJobById(ctx context.Context, id uuid.UUID) (*entities.Job, error)
	UpdateStatusJob(context context.Context, status constant.JobStatus, id uuid.UUID) error
	UpdateLessonVideoURL(ctx context.Context, lessonId uuid.UUID, url string) error
	GetRecordingsByLessonId(ctx context.Context, lessonId uuid.UUID) ([]*entities.Recording, error)
	GetRecordingChunksByLiveSessionId(ctx context.Context, liveSessionId uuid.UUID) ([]*entities.RecordingChunk, error)
	UpdateRecordingChunkStatus(ctx context.Context, chunkId uuid.UUID, status string) error
	UpdateLiveSessionRecording(ctx context.Context, liveSessionId uuid.UUID, recordingStatus string, finalVideoObjectName string, recordingDuration int, totalChunks int) error
}

type repo struct {
	db *gorm.DB
}

func (r *repo) UpdateLessonVideoURL(ctx context.Context, lessonId uuid.UUID, url string) error {
	lesson := &entities.Lesson{}
	err := r.GetDB().Model(lesson).Where("id = ?", lessonId).Update("video_url", url).Error
	if err != nil {
		return err
	}

	return nil
}

func (r *repo) FindJobById(ctx context.Context, id uuid.UUID) (*entities.Job, error) {
	job := &entities.Job{}
	err := r.GetDB().First(job, "id = ?", id).Error
	if err != nil {
		return nil, err
	}

	return job, nil
}

func (r *repo) UpdateStatusJob(context context.Context, status constant.JobStatus, id uuid.UUID) error {
	job := &entities.Job{}
	err := r.GetDB().First(job, "id = ?", id).Error
	if err != nil {
		return err
	}
	job.Status = status
	err = r.GetDB().Save(job).Error
	if err != nil {
		return err
	}
	return nil
}

func NewRepo(db *sql.DB) JobRepository {
	gormDB, _ := gorm.Open(postgres.New(postgres.Config{
		Conn: db}),
		&gorm.Config{
			Logger: logger.Default.LogMode(logger.Info),
		},
	)
	return &repo{
		db: gormDB,
	}
}

func (r *repo) GetDB() *gorm.DB {
	return r.db
}

func (r *repo) Transaction(ctx context.Context, callback func(ctx context.Context) error, opts ...*sql.TxOptions) error {
	return r.GetDB().Transaction(func(tx *gorm.DB) error {
		err := callback(ctx)
		if err != nil {
			return err
		}
		return nil
	}, opts...)
}

func (r *repo) GetRecordingsByLessonId(ctx context.Context, lessonId uuid.UUID) ([]*entities.Recording, error) {
	var recordings []*entities.Recording
	err := r.GetDB().Where("lesson_id = ?", lessonId).Order("chunk_number ASC").Find(&recordings).Error
	if err != nil {
		return nil, err
	}
	return recordings, nil
}

func (r *repo) GetRecordingChunksByLiveSessionId(ctx context.Context, liveSessionId uuid.UUID) ([]*entities.RecordingChunk, error) {
	var chunks []*entities.RecordingChunk
	err := r.GetDB().Where("live_session_id = ?", liveSessionId).Order("chunk_index ASC").Find(&chunks).Error
	if err != nil {
		return nil, err
	}
	return chunks, nil
}

func (r *repo) UpdateRecordingChunkStatus(ctx context.Context, chunkId uuid.UUID, status string) error {
	chunk := &entities.RecordingChunk{}
	err := r.GetDB().Model(chunk).Where("id = ?", chunkId).Update("status", status).Error
	if err != nil {
		return err
	}
	return nil
}

func (r *repo) UpdateLiveSessionRecording(ctx context.Context, liveSessionId uuid.UUID, recordingStatus string, finalVideoObjectName string, recordingDuration int, totalChunks int) error {
	liveSession := &entities.LiveSession{}
	updates := map[string]interface{}{
		"recording_status":       recordingStatus,
		"final_video_object_name": finalVideoObjectName,
		"recording_duration":      recordingDuration,
		"total_chunks":            totalChunks,
	}
	err := r.GetDB().Model(liveSession).Where("id = ?", liveSessionId).Updates(updates).Error
	if err != nil {
		return err
	}
	return nil
}
