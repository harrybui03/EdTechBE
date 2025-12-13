package main

import (
	"github.com/rs/zerolog/log"
	"os"
	"worker-transcode/cmd"
	"worker-transcode/config"
)

func main() {
	path, err := os.Getwd()
	if err != nil {
		log.Fatal().Err(err).Send()
	}
	cfg, err := config.Load(path)
	if err != nil {
		panic(err)
	}

	root := cmd.Root(cfg)
	if err := root.Execute(); err != nil {
		log.Fatal().Err(err).Send()
	}
}
