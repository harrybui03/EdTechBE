package main

import (
	"github.com/rs/zerolog/log"
	"worker-transcode/cmd"
	"worker-transcode/config"
)

func main() {
	cfg, err := config.Load(".env")
	if err != nil {
		panic(err)
	}

	root := cmd.Root(cfg)
	if err := root.Execute(); err != nil {
		log.Fatal().Err(err).Send()
	}
}
