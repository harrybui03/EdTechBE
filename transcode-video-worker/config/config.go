package config

import (
	"database/sql"
	"fmt"
	"os"
	"strconv"

	"github.com/joho/godotenv"
	_ "github.com/lib/pq"
	"github.com/minio/minio-go/v7"
	"github.com/minio/minio-go/v7/pkg/credentials"
)

type Config struct {
	MinIOBucket string
	App         App
	DB          *sql.DB
	Queue       *RabbitMQ
	Storage     *minio.Client
	Server      Server
}

type App struct {
	Environment string
	Host        string
	Protocol    string
}

type Server struct {
	HttpPort string
	Workers  int
}

type RabbitMQ struct {
	Host         string
	Port         int
	User         string
	Pass         string
	ExchangeName string
	Kind         string
}

func Load(path string) (*Config, error) {
	err := godotenv.Load(path)
	if err != nil {
		return nil, fmt.Errorf("error loading .env file from %s: %w", path, err)
	}

	psqlInfo := fmt.Sprintf("postgres://%s:%s@%s:%s/%s?sslmode=require",
		os.Getenv("POSTGRES_USER"),
		os.Getenv("POSTGRES_PASSWORD"),
		os.Getenv("DB_HOST"),
		os.Getenv("DB_PORT"),
		os.Getenv("POSTGRES_DB"),
	)

	db, err := sql.Open("postgres", psqlInfo)
	if err != nil {
		return nil, err
	}

	rabbitmqPort, err := strconv.Atoi(os.Getenv("RABBITMQ_PORT"))
	if err != nil {
		return nil, err
	}
	rabbitmq := &RabbitMQ{
		Host:         os.Getenv("RABBITMQ_HOST"),
		Port:         rabbitmqPort,
		User:         os.Getenv("RABBITMQ_USER"),
		Pass:         os.Getenv("RABBITMQ_PASS"),
		Kind:         os.Getenv("RABBITMQ_KIND"),
		ExchangeName: os.Getenv("RABBITMQ_EXCHANGE_NAME"),
	}

	minioClient, err := minio.New(os.Getenv("MINIO_URL"), &minio.Options{
		Creds:  credentials.NewStaticV4(os.Getenv("MINIO_ROOT_USER"), os.Getenv("MINIO_ROOT_PASSWORD"), ""),
		Secure: false,
	})
	if err != nil {
		return nil, err
	}

	workers, err := strconv.Atoi(os.Getenv("SERVER_WORKERS"))
	if err != nil {
		return nil, err
	}

	return &Config{
		MinIOBucket: os.Getenv("MINIO_BUCKET"),
		App: App{
			Environment: os.Getenv("APP_ENVIRONMENT"),
			Host:        os.Getenv("APP_HOST"),
			Protocol:    os.Getenv("APP_PROTOCOL"),
		},
		Server: Server{
			HttpPort: os.Getenv("WORKER_SERVER_PORT"),
			Workers:  workers,
		},
		DB:      db,
		Queue:   rabbitmq,
		Storage: minioClient,
	}, nil
}
