package rabbitmq

import (
	"context"
	"github.com/cenkalti/backoff/v5"
	amqp "github.com/rabbitmq/amqp091-go"
	"github.com/rs/zerolog"
	"sync"
	"time"
	"worker-transcode/config"
)

type RecordingConsumer[T any] interface {
	Consume(ctx context.Context, dependencies T) error
}

type recordingConsumer[T any] struct {
	conn       *amqp.Connection
	cfg        *config.RabbitMQ
	handler    func(ctx context.Context, msg amqp.Delivery, dependencies T) error
	numWorkers int
}

func (c recordingConsumer[T]) Consume(ctx context.Context, dependencies T) error {
	ch, err := c.conn.Channel()
	if err != nil {
		return err
	}
	defer ch.Close()

	exchangeName := "recording_exchange"
	queueName := "recording_merge_queue"
	routingKey := "recording.merge.request"
	dlxName := "transcoding_exchange_dlx"  // Dùng chung DLX với transcoding queue
	dlqName := "recording_merge_queue_dlq"
	dlqRoutingKey := "dlq.recording.merge.request"

	err = ch.ExchangeDeclare(exchangeName, c.cfg.Kind, true, false, false, false, nil)
	if err != nil {
		zerolog.Ctx(ctx).Error().Str("exchange", exchangeName).Msg("failed to declare exchange")
		return err
	}

	err = ch.ExchangeDeclare(dlxName, c.cfg.Kind, true, false, false, false, nil)
	if err != nil {
		zerolog.Ctx(ctx).Error().Str("exchange", dlxName).Msg("failed to declare dlx")
		return err
	}

	dlq, err := ch.QueueDeclare(dlqName, true, false, false, false, nil)
	if err != nil {
		zerolog.Ctx(ctx).Error().Str("queue", dlqName).Msg("failed to declare dlq")
		return err
	}

	err = ch.QueueBind(dlq.Name, dlqRoutingKey, dlxName, false, nil)
	if err != nil {
		zerolog.Ctx(ctx).Error().Msg("failed to bind dlq")
		return err
	}

	args := amqp.Table{
		"x-dead-letter-exchange":    dlxName,
		"x-dead-letter-routing-key": dlqRoutingKey,
	}
	q, err := ch.QueueDeclare(queueName, true, false, false, false, args)
	if err != nil {
		zerolog.Ctx(ctx).Error().Str("queue", queueName).Msg("failed to declare queue")
		return err
	}

	err = ch.QueueBind(q.Name, routingKey, exchangeName, false, nil)
	if err != nil {
		zerolog.Ctx(ctx).Error().Str("queue", queueName).Msg("failed to bind queue")
		return err
	}

	err = ch.Qos(c.numWorkers, 0, false)
	if err != nil {
		zerolog.Ctx(ctx).Error().Str("queue", queueName).Msg("failed to set QoS")
		return err
	}

	deliveries, err := ch.Consume(queueName, "", false, false, false, false, nil)
	if err != nil {
		zerolog.Ctx(ctx).Error().Str("queue", queueName).Msg("failed to consume queue")
		return err
	}

	zerolog.Ctx(ctx).Info().
		Str("queue", queueName).
		Str("exchange", exchangeName).
		Str("routing_key", routingKey).
		Int("workers", c.numWorkers).
		Msg("recording merge consumer started")

	jobs := make(chan amqp.Delivery, c.numWorkers)
	var wg sync.WaitGroup
	for i := 1; i <= c.numWorkers; i++ {
		wg.Add(1)
		go func(workerId int) {
			defer wg.Done()
			for msg := range jobs {
				operation := func() (string, error) {
					err := c.handler(ctx, msg, dependencies)
					if err != nil {
						return "", err
					}
					return "", nil
				}

				bo := backoff.NewExponentialBackOff()
				bo.MaxInterval = 10 * time.Second

				_, err := backoff.Retry(ctx, operation, backoff.WithBackOff(bo), backoff.WithMaxTries(5))
				if err != nil {
					zerolog.Ctx(ctx).Error().Err(err).Int("worker_id", workerId).Msg("failed to handle message after all retries")
					if nackErr := msg.Nack(false, false); nackErr != nil {
						zerolog.Ctx(ctx).Error().Err(nackErr).Msg("failed to nack message to send to DLQ")
					}
				} else {
					if ackErr := msg.Ack(false); ackErr != nil {
						zerolog.Ctx(ctx).Error().Err(ackErr).Msg("failed to acknowledge message")
					}
				}
			}
		}(i)
	}

	for {
		select {
		case delivery, ok := <-deliveries:
			if !ok {
				close(jobs)
				wg.Wait()
				return nil
			}

			jobs <- delivery
		case <-ctx.Done():
			close(jobs)
			wg.Wait()
			return ctx.Err()
		}
	}
}

func NewRecordingConsumer[T any](
	conn *amqp.Connection,
	cfg *config.RabbitMQ,
	numWorkers int,
	handler func(ctx context.Context, msg amqp.Delivery, dependencies T) error,
) RecordingConsumer[T] {
	if numWorkers < 1 {
		numWorkers = 1
	}
	return &recordingConsumer[T]{
		conn:       conn,
		cfg:        cfg,
		handler:    handler,
		numWorkers: numWorkers,
	}
}

