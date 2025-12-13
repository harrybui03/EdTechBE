package config

import (
	"context"
	"fmt"
	"github.com/cenkalti/backoff/v5"
	amqp "github.com/rabbitmq/amqp091-go"
	"github.com/rs/zerolog"
	"time"
)

func NewRabbitMQConn(ctx context.Context, cfg *RabbitMQ) (*amqp.Connection, error) {
	connAddr := fmt.Sprintf("amqp://%s:%s@%s:%d/", cfg.User, cfg.Pass, cfg.Host, cfg.Port)

	operation := func() (*amqp.Connection, error) {
		conn, err := amqp.Dial(connAddr)
		if err != nil {
			zerolog.Ctx(ctx).Error().Err(err).Msg("Failed to connect to RabbitMQ. Retrying...")
			return nil, err
		}

		return conn, nil
	}

	bo := backoff.NewExponentialBackOff()
	bo.MaxInterval = 10 * time.Second
	maxRetries := uint(5)
	conn, err := backoff.Retry(ctx, operation, backoff.WithBackOff(bo), backoff.WithMaxTries(maxRetries))
	if err != nil {
		zerolog.Ctx(ctx).Error().Err(err).Msg("Failed to connect to RabbitMQ. Retrying...")
		return nil, err
	}

	zerolog.Ctx(ctx).Info().Msg("Successfully connected to RabbitMQ")
	go func() {
		select {
		case <-ctx.Done():
			err := conn.Close()
			if err != nil {
				zerolog.Ctx(ctx).Error().Err(err).Msg("Failed to close RabbitMQ connection")
			}
			zerolog.Ctx(ctx).Info().Msg("RabbitMQ connection closed")
		}
	}()

	return conn, nil
}
