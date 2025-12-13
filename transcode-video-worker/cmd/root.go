package cmd

import (
	"github.com/spf13/cobra"
	"worker-transcode/config"
)

func Root(config *config.Config) *cobra.Command {
	rootCmd := &cobra.Command{}
	rootCmd.AddCommand(server(config))
	return rootCmd
}
