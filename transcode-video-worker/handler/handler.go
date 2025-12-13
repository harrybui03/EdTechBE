package handler

import (
	"context"
	"encoding/json"
	amqp "github.com/rabbitmq/amqp091-go"
	"github.com/rs/zerolog"
	"worker-transcode/dto"
	"worker-transcode/service"
)

type ServiceDependencies struct {
	TranscodeService      service.Service
	RecordingMergeService service.RecordingMergeService
}

func JobHandler(ctx context.Context, msg amqp.Delivery, deps ServiceDependencies) error {
	var job dto.JobMessage
	if err := json.Unmarshal(msg.Body, &job); err != nil {
		return err
	}

	err := deps.TranscodeService.Process(ctx, job)
	if err != nil {
		return err
	}

	return nil
}

func RecordingMergeHandler(ctx context.Context, msg amqp.Delivery, deps ServiceDependencies) error {
	var recordingMsg dto.RecordingMergeMessage
	if err := json.Unmarshal(msg.Body, &recordingMsg); err != nil {
		zerolog.Ctx(ctx).Error().Err(err).Msg("failed to unmarshal recording merge message")
		return err
	}

	zerolog.Ctx(ctx).Info().
		Str("job_id", recordingMsg.JobId.String()).
		Str("live_session_id", recordingMsg.LiveSessionId.String()).
		Msg("received recording merge message")

	err := deps.RecordingMergeService.ProcessRecordingMerge(ctx, recordingMsg)
	if err != nil {
		return err
	}

	return nil
}
