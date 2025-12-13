package cmd

import (
	"github.com/spf13/cobra"
	"worker-transcode/config"
	server2 "worker-transcode/server"
)

func server(config *config.Config) *cobra.Command {
	return &cobra.Command{
		Use:   "server",
		Short: "start http server",
		Run: func(cmd *cobra.Command, args []string) {
			server2.RunHttp(config)
		},
	}
}
