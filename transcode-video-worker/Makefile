.PHONY: install install-ffmpeg

# Installs all required tools (Go and system dependencies)
install: install-ffmpeg


# Installs FFmpeg using apt-get for Debian/Ubuntu systems
install-ffmpeg:
	@echo "--> Installing FFmpeg..."
	@sudo apt-get update && sudo apt-get install -y ffmpeg
create-config-file:
	cp config.tmp.yaml config.yaml
run:
	@go run main.go server