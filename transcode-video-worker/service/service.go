package service

import (
	"context"
	"errors"
	"github.com/minio/minio-go/v7"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
	"os"
	"path/filepath"
	"strings"
	"worker-transcode/config"
	"worker-transcode/constant"
	"worker-transcode/dto"
	"worker-transcode/repository"
)

var ErrNonRetryable = errors.New("non-retryable error")

type Service interface {
	Process(ctx context.Context, message dto.JobMessage) error
}

type service struct {
	repo repository.JobRepository
	cfg  *config.Config
}

func (s service) Process(ctx context.Context, message dto.JobMessage) (err error) {
	zerolog.Ctx(ctx).Info().Str("job_id", message.JobId.String()).Msg("processing job")
	path := filepath.Dir(message.ObjectPath)
	fileName := filepath.Base(message.ObjectPath)
	job, err := s.repo.FindJobById(ctx, message.JobId)
	if err != nil {
		zerolog.Ctx(ctx).Error().Err(err).Msg("failed to find job by id")
		return err
	}

	if job.Status != constant.JobStatusPending {
		zerolog.Ctx(ctx).Info().Str("job_id", message.JobId.String()).Msg("job is not pending")
		return nil
	}

	if err := s.repo.UpdateStatusJob(ctx, constant.JobStatusProcessing, message.JobId); err != nil {
		log.Error().Err(err).Msg("failed to update job status")
		return err
	}

	defer func() {
		if err != nil {
			if errors.Is(err, ErrNonRetryable) {
				if updateErr := s.repo.UpdateStatusJob(ctx, constant.JobStatusFailed, message.JobId); updateErr != nil {
					log.Error().Err(updateErr).Msg("failed to update job status")
				}
				err = nil
			} else {
				if updateErr := s.repo.UpdateStatusJob(ctx, constant.JobStatusPending, message.JobId); updateErr != nil {
					log.Error().Err(updateErr).Msg("failed to update job status")
				}
			}
		}
	}()

	tempDir := filepath.Join("temp", message.JobId.String())
	defer os.RemoveAll(tempDir)

	inputDir := filepath.Join(tempDir, "input")
	outputDir := filepath.Join(tempDir, "output")

	if err = os.MkdirAll(inputDir, os.ModePerm); err != nil {
		zerolog.Ctx(ctx).Error().Err(err).Msg("failed to create input directory")
		return errors.Join(ErrNonRetryable, err)
	}
	if err = os.MkdirAll(outputDir, os.ModePerm); err != nil {
		zerolog.Ctx(ctx).Error().Err(err).Msg("failed to create output dir")
		return errors.Join(ErrNonRetryable, err)
	}

	inputFilepath := filepath.Join(inputDir, fileName)
	zerolog.Ctx(ctx).Info().Str("input_file", inputFilepath).Msg("downloading input file")
	err = s.cfg.Storage.FGetObject(ctx, s.cfg.MinIOBucket, message.ObjectPath, inputFilepath, minio.GetObjectOptions{})
	if err != nil {
		zerolog.Ctx(ctx).Error().Err(err).Msg("failed to download file")
		return err
	}

	zerolog.Ctx(ctx).Info().Msg("transcode file")
	if err = transcodeToHLS(inputFilepath, outputDir); err != nil {
		zerolog.Ctx(ctx).Error().Err(err).Msg("failed to transcode file")
		return errors.Join(ErrNonRetryable, err)
	}

	if err = createMasterPlaylist(outputDir); err != nil {
		zerolog.Ctx(ctx).Error().Err(err).Msg("failed to create master playlist")
		return errors.Join(ErrNonRetryable, err)
	}

	zerolog.Ctx(ctx).Info().Msg("upload transcode file")
	err = uploadDirectory(ctx, s.cfg.Storage, s.cfg.MinIOBucket, outputDir, path)
	if err != nil {
		zerolog.Ctx(ctx).Error().Err(err).Msg("failed to upload directory")
		return err
	}

	zerolog.Ctx(ctx).Info().Msg("deleting original file")
	err = s.cfg.Storage.RemoveObject(ctx, s.cfg.MinIOBucket, message.ObjectPath, minio.RemoveObjectOptions{})
	if err != nil {
		zerolog.Ctx(ctx).Error().Err(err).Msg("failed to delete original file")
		return err
	}

	if err = s.repo.UpdateStatusJob(ctx, constant.JobStatusCompleted, message.JobId); err != nil {
		zerolog.Ctx(ctx).Error().Err(err).Msg("failed to update job status")
		return err
	}

	if err = s.repo.UpdateLessonVideoURL(ctx, job.EntityId, filepath.Join(path, "master.m3u8")); err != nil {
		zerolog.Ctx(ctx).Error().Err(err).Msg("failed to update lesson video url")
		return err
	}

	zerolog.Ctx(ctx).Info().Str("job_id", message.JobId.String()).Msg("job completed")

	return nil
}

func uploadDirectory(ctx context.Context, client *minio.Client, bucket, localPath, remotePrefix string) error {
	return filepath.Walk(localPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if info.IsDir() {
			return nil
		}

		relativePath, err := filepath.Rel(localPath, path)
		if err != nil {
			return err
		}

		objectName := filepath.Join(remotePrefix, relativePath)

		objectName = strings.ReplaceAll(objectName, "\\", "/")

		_, uploadErr := client.FPutObject(ctx, bucket, objectName, path, minio.PutObjectOptions{})
		return uploadErr
	})
}

func NewService(repo repository.JobRepository, cfg *config.Config) Service {
	return &service{
		repo: repo,
		cfg:  cfg,
	}
}
