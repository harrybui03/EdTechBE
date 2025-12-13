package server

import (
	"context"
	"errors"
	"fmt"
	"github.com/gin-gonic/gin"
	"github.com/rs/zerolog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"
	"worker-transcode/config"
	"worker-transcode/constant"
	jobHandler "worker-transcode/handler"
	"worker-transcode/pkg/rabbitmq"
	"worker-transcode/repository"
	"worker-transcode/service"
)

func RunHttp(cfg *config.Config) {
	ctx, cancel := signal.NotifyContext(setupLogger(cfg), syscall.SIGINT, syscall.SIGTERM)
	defer cancel()

	zerolog.Ctx(ctx).Info().Str("env", cfg.App.Environment).Bool("isProduction", cfg.App.Environment == constant.EnvironmentProduction.String()).Send()
	if cfg.App.Environment == constant.EnvironmentProduction.String() {
		gin.SetMode(gin.ReleaseMode)
	}

	conn, err := config.NewRabbitMQConn(ctx, cfg.Queue)
	if err != nil {
		zerolog.Ctx(ctx).Error().Err(err).Msg("NewRabbitMQConn")
	}

	repo := repository.NewRepo(cfg.DB)
	transcodeService := service.NewService(repo, cfg)
	recordingMergeService := service.NewRecordingMergeService(repo, cfg)

	serviceDeps := jobHandler.ServiceDependencies{
		TranscodeService:      transcodeService,
		RecordingMergeService: recordingMergeService,
	}

	// Start transcoding consumer
	transcodeConsumer := rabbitmq.NewConsumer(conn, cfg.Queue, cfg.Server.Workers, jobHandler.JobHandler)
	go func() {
		err := transcodeConsumer.Consume(ctx, serviceDeps)
		if err != nil {
			zerolog.Ctx(ctx).Error().Err(err).Msg("Transcode consumer error")
		}
	}()

	// Start recording merge consumer
	recordingConsumer := rabbitmq.NewRecordingConsumer(conn, cfg.Queue, cfg.Server.Workers, jobHandler.RecordingMergeHandler)
	go func() {
		err := recordingConsumer.Consume(ctx, serviceDeps)
		if err != nil {
			zerolog.Ctx(ctx).Error().Err(err).Msg("Recording merge consumer error")
		}
	}()

	r := gin.Default()
	addHealth(r)

	handler := http.Server{
		Handler:           r,
		Addr:              fmt.Sprintf(":%s", cfg.Server.HttpPort),
		ReadHeaderTimeout: 10 * time.Second,
	}

	go func() {
		zerolog.Ctx(ctx).Info().Str("env", cfg.App.Environment).Msg("start http server")
		if err := handler.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			zerolog.Ctx(ctx).Error().Str("env", cfg.App.Environment).Msg(err.Error())
		}
	}()

	<-ctx.Done()
	zerolog.Ctx(ctx).Info().Msg("shutting down server")
	if err := handler.Shutdown(ctx); err != nil {
		zerolog.Ctx(ctx).Error().Str("env", cfg.App.Environment).Msg(err.Error())
	}

	zerolog.Ctx(ctx).Info().Str("env", cfg.App.Environment).Msg("server shutdown")
}

func addHealth(r *gin.Engine) {
	r.GET("/health", func(c *gin.Context) {
		c.JSON(200, gin.H{
			"status": "ok",
		})
	})
}

func setupLogger(cfg *config.Config) context.Context {
	zerolog.SetGlobalLevel(zerolog.InfoLevel)
	if cfg.App.Environment == constant.EnvironmentDevelop.String() {
		zerolog.SetGlobalLevel(zerolog.DebugLevel)
	}

	// Log to standard output
	logger := zerolog.New(os.Stdout).With().Timestamp().Logger()
	ctx := logger.WithContext(context.Background())

	return ctx
}
