package service

import (
	"context"
	"errors"
	"fmt"
	"github.com/minio/minio-go/v7"
	"github.com/rs/zerolog"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"worker-transcode/config"
	"worker-transcode/constant"
	"worker-transcode/dto"
	"worker-transcode/entities"
	"worker-transcode/repository"
)

type RecordingMergeService interface {
	ProcessRecordingMerge(ctx context.Context, message dto.RecordingMergeMessage) error
}

type recordingMergeService struct {
	repo repository.JobRepository
	cfg  *config.Config
}

func (s *recordingMergeService) ProcessRecordingMerge(ctx context.Context, message dto.RecordingMergeMessage) (err error) {
	zerolog.Ctx(ctx).Info().
		Str("job_id", message.JobId.String()).
		Str("live_session_id", message.LiveSessionId.String()).
		Msg("processing recording merge job")

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
		zerolog.Ctx(ctx).Error().Err(err).Msg("failed to update job status")
		return err
	}

	defer func() {
		if err != nil {
			if errors.Is(err, ErrNonRetryable) {
				if updateErr := s.repo.UpdateStatusJob(ctx, constant.JobStatusFailed, message.JobId); updateErr != nil {
					zerolog.Ctx(ctx).Error().Err(updateErr).Msg("failed to update job status")
				}
				err = nil
			} else {
				if updateErr := s.repo.UpdateStatusJob(ctx, constant.JobStatusPending, message.JobId); updateErr != nil {
					zerolog.Ctx(ctx).Error().Err(updateErr).Msg("failed to update job status")
				}
			}
		}
	}()

	// Get recording chunks from database
	zerolog.Ctx(ctx).Info().Msg("fetching recording chunks from database")
	chunks, err := s.repo.GetRecordingChunksByLiveSessionId(ctx, message.LiveSessionId)
	if err != nil {
		zerolog.Ctx(ctx).Error().Err(err).Msg("failed to get recording chunks from database")
		return err
	}

	if len(chunks) == 0 {
		err = fmt.Errorf("no recording chunks found for live_session_id: %s", message.LiveSessionId.String())
		zerolog.Ctx(ctx).Error().Err(err).Msg("no chunks to merge")
		return errors.Join(ErrNonRetryable, err)
	}

	zerolog.Ctx(ctx).Info().Int("chunk_count", len(chunks)).Msg("found recording chunks in database")

	// Log danh sách chunks từ database
	chunkList := make([]string, 0, len(chunks))
	for _, chunk := range chunks {
		chunkList = append(chunkList, fmt.Sprintf("chunk_index=%d, object_name=%s, size=%d bytes", 
			chunk.ChunkIndex, chunk.ObjectName, func() int64 {
				if chunk.FileSize != nil {
					return *chunk.FileSize
				}
				return 0
			}()))
	}
	zerolog.Ctx(ctx).Info().
		Strs("chunks", chunkList).
		Msg("chunks to download and merge")

	// Update chunks status to PROCESSING
	for _, chunk := range chunks {
		if err := s.repo.UpdateRecordingChunkStatus(ctx, chunk.ID, "PROCESSING"); err != nil {
			zerolog.Ctx(ctx).Warn().Err(err).Str("chunk_id", chunk.ID.String()).Msg("failed to update chunk status to PROCESSING")
		}
	}

	// Create temporary directories
	tempDir := filepath.Join("temp", message.JobId.String())
	defer os.RemoveAll(tempDir)

	chunksDir := filepath.Join(tempDir, "chunks")
	outputDir := filepath.Join(tempDir, "output")

	if err = os.MkdirAll(chunksDir, os.ModePerm); err != nil {
		zerolog.Ctx(ctx).Error().Err(err).Msg("failed to create chunks directory")
		return errors.Join(ErrNonRetryable, err)
	}
	if err = os.MkdirAll(outputDir, os.ModePerm); err != nil {
		zerolog.Ctx(ctx).Error().Err(err).Msg("failed to create output directory")
		return errors.Join(ErrNonRetryable, err)
	}

	// Download all chunks from MinIO using object_name from database
	zerolog.Ctx(ctx).Info().Int("total_chunks", len(chunks)).Msg("starting to download chunks from MinIO")
	chunkPaths, err := s.downloadChunks(ctx, chunks, chunksDir)
	if err != nil {
		zerolog.Ctx(ctx).Error().Err(err).Msg("failed to download chunks")
		return err
	}

	if len(chunkPaths) == 0 {
		err = fmt.Errorf("no chunks downloaded")
		zerolog.Ctx(ctx).Error().Err(err).Msg("no chunks to merge")
		return errors.Join(ErrNonRetryable, err)
	}

	zerolog.Ctx(ctx).Info().
		Int("chunk_count", len(chunkPaths)).
		Strs("downloaded_files", chunkPaths).
		Msg("all chunks downloaded successfully")

	// Merge chunks using FFmpeg
	outputFileName := "final.mp4"
	outputFilePath := filepath.Join(outputDir, outputFileName)

	zerolog.Ctx(ctx).Info().
		Int("chunk_count", len(chunkPaths)).
		Strs("chunks_to_merge", chunkPaths).
		Str("output_file", outputFilePath).
		Msg("starting to merge chunks with FFmpeg")
	
	if err = mergeWebMChunks(ctx, chunkPaths, outputFilePath); err != nil {
		zerolog.Ctx(ctx).Error().Err(err).Msg("failed to merge chunks")
		// Update chunks status to FAILED
		for _, chunk := range chunks {
			s.repo.UpdateRecordingChunkStatus(ctx, chunk.ID, "FAILED")
		}
		return errors.Join(ErrNonRetryable, err)
	}

	zerolog.Ctx(ctx).Info().
		Str("output_file", outputFilePath).
		Int("merged_chunks", len(chunkPaths)).
		Msg("chunks merged successfully with FFmpeg")

	// Build output path: live-recordings/{sessionId}/final/recording.mp4
	// Get session folder from chunk path (e.g., live-recordings/{sessionId}/chunks/chunk_0000.webm -> live-recordings/{sessionId})
	chunkFolder := filepath.Dir(chunks[0].ObjectName) // live-recordings/{sessionId}/chunks
	sessionFolder := filepath.Dir(chunkFolder)        // live-recordings/{sessionId}
	outputKey := filepath.Join(sessionFolder, "final", "recording.mp4")
	outputKey = strings.ReplaceAll(outputKey, "\\", "/")

	zerolog.Ctx(ctx).Info().Str("output_key", outputKey).Msg("uploading final video to MinIO")
	_, err = s.cfg.Storage.FPutObject(ctx, s.cfg.MinIOBucket, outputKey, outputFilePath, minio.PutObjectOptions{
		ContentType: "video/mp4",
	})
	if err != nil {
		zerolog.Ctx(ctx).Error().Err(err).Msg("failed to upload final video")
		return err
	}

	// Update chunks status to COMPLETED
	for _, chunk := range chunks {
		if err := s.repo.UpdateRecordingChunkStatus(ctx, chunk.ID, "COMPLETED"); err != nil {
			zerolog.Ctx(ctx).Warn().Err(err).Str("chunk_id", chunk.ID.String()).Msg("failed to update chunk status to COMPLETED")
		}
	}

	// Calculate total recording duration from chunks
	totalDuration := 0
	for _, chunk := range chunks {
		if chunk.DurationSeconds != nil {
			totalDuration += *chunk.DurationSeconds
		}
	}

	// Update live_sessions with recording information
	zerolog.Ctx(ctx).Info().
		Str("live_session_id", message.LiveSessionId.String()).
		Str("final_video_object_name", outputKey).
		Int("recording_duration", totalDuration).
		Int("total_chunks", len(chunks)).
		Msg("updating live_sessions with recording information")

	if err = s.repo.UpdateLiveSessionRecording(ctx, message.LiveSessionId, "COMPLETED", outputKey, totalDuration, len(chunks)); err != nil {
		zerolog.Ctx(ctx).Error().Err(err).Msg("failed to update live_sessions")
		// Don't return error, just log it - job is still successful
	} else {
		zerolog.Ctx(ctx).Info().
			Str("live_session_id", message.LiveSessionId.String()).
			Msg("live_sessions updated successfully")
	}

	// Update job status to completed
	if err = s.repo.UpdateStatusJob(ctx, constant.JobStatusCompleted, message.JobId); err != nil {
		zerolog.Ctx(ctx).Error().Err(err).Msg("failed to update job status")
		return err
	}

	zerolog.Ctx(ctx).Info().
		Str("job_id", message.JobId.String()).
		Str("live_session_id", message.LiveSessionId.String()).
		Str("output_key", outputKey).
		Int("chunks_processed", len(chunks)).
		Int("recording_duration_seconds", totalDuration).
		Msg("recording merge job completed successfully")

	return nil
}

func (s *recordingMergeService) downloadChunks(ctx context.Context, chunks []*entities.RecordingChunk, localDir string) ([]string, error) {
	var chunkPaths []string

	zerolog.Ctx(ctx).Info().
		Int("total_chunks", len(chunks)).
		Msg("beginning download process for all chunks")

	for i, chunk := range chunks {
		// Download chunk using object_name from database
		objectName := chunk.ObjectName
		
		// Create local filename based on chunk_index to maintain order
		fileName := fmt.Sprintf("chunk-%03d.webm", chunk.ChunkIndex)
		localPath := filepath.Join(localDir, fileName)
		
		zerolog.Ctx(ctx).Info().
			Int("chunk_number", i+1).
			Int("total_chunks", len(chunks)).
			Int("chunk_index", chunk.ChunkIndex).
			Str("object_name", objectName).
			Str("local_path", localPath).
			Interface("file_size", chunk.FileSize).
			Msg("downloading chunk from MinIO")

		err := s.cfg.Storage.FGetObject(ctx, s.cfg.MinIOBucket, objectName, localPath, minio.GetObjectOptions{})
		if err != nil {
			zerolog.Ctx(ctx).Error().
				Err(err).
				Int("chunk_index", chunk.ChunkIndex).
				Str("object_name", objectName).
				Msg("failed to download chunk")
			return nil, fmt.Errorf("failed to download chunk %s (chunk_index: %d): %w", objectName, chunk.ChunkIndex, err)
		}

		// Get file size after download
		fileInfo, statErr := os.Stat(localPath)
		fileSize := int64(0)
		if statErr == nil {
			fileSize = fileInfo.Size()
		}

		zerolog.Ctx(ctx).Info().
			Int("chunk_index", chunk.ChunkIndex).
			Str("local_path", localPath).
			Int64("downloaded_size_bytes", fileSize).
			Msg("chunk downloaded successfully")

		chunkPaths = append(chunkPaths, localPath)
	}

	zerolog.Ctx(ctx).Info().
		Int("downloaded_count", len(chunkPaths)).
		Strs("downloaded_files", chunkPaths).
		Msg("all chunks downloaded successfully")

	return chunkPaths, nil
}

func mergeWebMChunks(ctx context.Context, chunkPaths []string, outputPath string) error {
	if len(chunkPaths) == 0 {
		return fmt.Errorf("no chunks to merge")
	}

	zerolog.Ctx(ctx).Info().
		Int("chunk_count", len(chunkPaths)).
		Strs("input_chunks", chunkPaths).
		Str("output_path", outputPath).
		Msg("preparing to merge chunks - converting WebM to MP4 first")

	// Step 1: Convert each WebM chunk to MP4
	tempDir := filepath.Dir(outputPath)
	mp4Chunks := make([]string, 0, len(chunkPaths))
	
	for i, webmPath := range chunkPaths {
		// Create temp MP4 file name
		mp4Path := filepath.Join(tempDir, fmt.Sprintf("chunk-%03d.mp4", i))
		mp4Chunks = append(mp4Chunks, mp4Path)
		
		zerolog.Ctx(ctx).Info().
			Int("chunk_index", i+1).
			Int("total_chunks", len(chunkPaths)).
			Str("webm_path", webmPath).
			Str("mp4_path", mp4Path).
			Msg("converting WebM chunk to MP4")

		// Convert WebM to MP4
		convertArgs := []string{
			"-i", webmPath,
			"-c:v", "libx264",    // H.264 video
			"-preset", "medium",
			"-crf", "23",
			"-c:a", "aac",        // AAC audio
			"-b:a", "128k",
			"-movflags", "+faststart",
			"-y",
			mp4Path,
		}

		cmd := exec.Command("ffmpeg", convertArgs...)
		output, err := cmd.CombinedOutput()
		
		if err != nil {
			zerolog.Ctx(ctx).Error().
				Err(err).
				Str("webm_path", webmPath).
				Str("mp4_path", mp4Path).
				Str("ffmpeg_output", string(output)).
				Msg("failed to convert WebM chunk to MP4")
			
			// Cleanup already converted MP4 files on error
			for _, mp4Path := range mp4Chunks {
				os.Remove(mp4Path)
			}
			
			return fmt.Errorf("failed to convert chunk %d: %w\nOutput: %s", i, err, string(output))
		}

		// Get converted file size
		fileInfo, statErr := os.Stat(mp4Path)
		mp4Size := int64(0)
		if statErr == nil {
			mp4Size = fileInfo.Size()
		}

		zerolog.Ctx(ctx).Info().
			Int("chunk_index", i+1).
			Str("mp4_path", mp4Path).
			Int64("mp4_size_bytes", mp4Size).
			Msg("WebM chunk converted to MP4 successfully")
	}

	// Step 2: Create concat file for MP4 files
	concatFilePath := filepath.Join(tempDir, "concat_list.txt")
	defer os.Remove(concatFilePath)

	var concatContent strings.Builder
	var concatLines []string
	for i, mp4Path := range mp4Chunks {
		absPath, err := filepath.Abs(mp4Path)
		if err != nil {
			// Cleanup on error
			for _, mp4Path := range mp4Chunks {
				os.Remove(mp4Path)
			}
			return fmt.Errorf("failed to get absolute path: %w", err)
		}
		
		escapedPath := strings.ReplaceAll(absPath, "'", "'\\''")
		line := fmt.Sprintf("file '%s'", escapedPath)
		concatContent.WriteString(line + "\n")
		concatLines = append(concatLines, line)
		
		zerolog.Ctx(ctx).Debug().
			Int("chunk_order", i+1).
			Str("mp4_path", mp4Path).
			Str("absolute_path", absPath).
			Str("concat_line", line).
			Msg("added MP4 chunk to concat list")
	}

	concatFileContent := concatContent.String()
	if err := os.WriteFile(concatFilePath, []byte(concatFileContent), 0644); err != nil {
		// Cleanup on error
		for _, mp4Path := range mp4Chunks {
			os.Remove(mp4Path)
		}
		return fmt.Errorf("failed to create concat file: %w", err)
	}

	zerolog.Ctx(ctx).Info().
		Str("concat_file", concatFilePath).
		Int("total_chunks", len(mp4Chunks)).
		Strs("concat_lines", concatLines).
		Str("concat_file_content", concatFileContent).
		Msg("concat file created for MP4 chunks, starting merge")

	// Step 3: Merge MP4 files (can use copy since all are same format now)
	ffmpegArgs := []string{
		"-f", "concat",
		"-safe", "0",
		"-i", concatFilePath,
		"-c", "copy",  // Can use copy since all are MP4 with same codec
		"-movflags", "+faststart",
		"-y",
		outputPath,
	}

	zerolog.Ctx(ctx).Info().
		Strs("ffmpeg_args", ffmpegArgs).
		Msg("executing FFmpeg merge command for MP4 files")

	cmd := exec.Command("ffmpeg", ffmpegArgs...)
	output, err := cmd.CombinedOutput()
	
	zerolog.Ctx(ctx).Info().
		Str("ffmpeg_output", string(output)).
		Msg("FFmpeg merge output")

	if err != nil {
		zerolog.Ctx(ctx).Error().
			Err(err).
			Str("ffmpeg_output", string(output)).
			Msg("FFmpeg merge failed")
		
		// Cleanup temp MP4 files on error
		for _, mp4Path := range mp4Chunks {
			os.Remove(mp4Path)
		}
		
		return fmt.Errorf("ffmpeg merge failed: %w\nOutput: %s", err, string(output))
	}

	// Step 4: Cleanup temp MP4 files
	zerolog.Ctx(ctx).Info().Msg("cleaning up temporary MP4 files")
	for _, mp4Path := range mp4Chunks {
		if err := os.Remove(mp4Path); err != nil {
			zerolog.Ctx(ctx).Warn().Err(err).Str("mp4_path", mp4Path).Msg("failed to remove temp MP4 file")
		} else {
			zerolog.Ctx(ctx).Debug().Str("mp4_path", mp4Path).Msg("removed temp MP4 file")
		}
	}

	// Get output file size
	fileInfo, statErr := os.Stat(outputPath)
	outputSize := int64(0)
	if statErr == nil {
		outputSize = fileInfo.Size()
	}

	zerolog.Ctx(ctx).Info().
		Str("output_path", outputPath).
		Int64("output_size_bytes", outputSize).
		Int("merged_chunks", len(chunkPaths)).
		Msg("FFmpeg merge completed successfully")

	return nil
}

func NewRecordingMergeService(repo repository.JobRepository, cfg *config.Config) RecordingMergeService {
	return &recordingMergeService{
		repo: repo,
		cfg:  cfg,
	}
}

