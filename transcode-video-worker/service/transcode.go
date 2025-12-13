package service

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

type Resolution struct {
	Width     int
	Height    int
	Bitrate   string // e.g., "800k"
	AudioRate string // e.g., "96k"
}

// Define the target resolutions for HLS.
var resolutions = []Resolution{
	{Width: 256, Height: 144, Bitrate: "200k", AudioRate: "64k"},
	{Width: 640, Height: 360, Bitrate: "800k", AudioRate: "96k"},
	{Width: 854, Height: 480, Bitrate: "1500k", AudioRate: "128k"},
	{Width: 1280, Height: 720, Bitrate: "3000k", AudioRate: "192k"},
	{Width: 1920, Height: 1080, Bitrate: "5000k", AudioRate: "192k"},
}

func transcodeToHLS(inputFilepath, outputDir string) error {
	var filterComplexBuilder strings.Builder
	for _, r := range resolutions {
		filterComplexBuilder.WriteString(
			fmt.Sprintf("[0:v]scale=w=%d:h=%d:force_original_aspect_ratio=decrease,pad=w=%d:h=%d:x=(ow-iw)/2:y=(oh-ih)/2[v%d]; ",
				r.Width, r.Height, r.Width, r.Height, r.Height))
	}

	ffmpegArgs := []string{
		"-i", inputFilepath,
		"-filter_complex", strings.TrimSuffix(filterComplexBuilder.String(), "; "),
	}

	for _, r := range resolutions {

		playlistName := fmt.Sprintf("%dp.m3u8", r.Height)
		segmentName := fmt.Sprintf("%dp_%%03d.ts", r.Height)

		ffmpegArgs = append(ffmpegArgs,
			"-map", fmt.Sprintf("[v%d]", r.Height),

			"-c:v", "libx264",
			"-preset", "veryfast",
			"-crf", "22", // Constant Rate Factor for quality
			"-b:v", r.Bitrate,
			"-maxrate", r.Bitrate,
			"-bufsize", r.Bitrate,

			"-f", "hls",
			"-hls_time", "6",
			"-hls_playlist_type", "vod",
			"-hls_segment_filename", filepath.Join(outputDir, segmentName),
			filepath.Join(outputDir, playlistName),
		)
	}

	highestAudioRate := "96k" // Default
	if len(resolutions) > 0 {
		highestAudioRate = resolutions[len(resolutions)-1].AudioRate
	}
	ffmpegArgs = append(ffmpegArgs,
		"-map", "0:a:0?",
		"-c:a", "aac",
		"-b:a", highestAudioRate,
		"-f", "hls",
		"-hls_time", "6",
		"-hls_playlist_type", "vod",
		"-hls_segment_filename", filepath.Join(outputDir, "audio_%03d.ts"),
		filepath.Join(outputDir, "audio.m3u8"))

	cmd := exec.Command("ffmpeg", ffmpegArgs...)
	log.Printf("Executing FFmpeg command: ffmpeg %s", strings.Join(ffmpegArgs, " "))

	output, err := cmd.CombinedOutput()
	if err != nil {
		log.Printf("FFmpeg output:\n%s\n", string(output))
		return fmt.Errorf("ffmpeg execution failed: %w", err)
	}

	return nil
}

func createMasterPlaylist(outputDir string) error {
	masterPlaylistPath := filepath.Join(outputDir, "master.m3u8")
	var contentBuilder strings.Builder
	contentBuilder.WriteString("#EXTM3U\n")
	contentBuilder.WriteString("#EXT-X-VERSION:3\n\n")

	contentBuilder.WriteString(`#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID="audio",NAME="English",DEFAULT=YES,AUTOSELECT=YES,URI="audio.m3u8"` + "\n\n")

	log.Println("Creating master playlist...")

	for _, r := range resolutions {
		var videoBitrateBPS int
		fmt.Sscanf(r.Bitrate, "%dk", &videoBitrateBPS)

		var audioBitrateBPS int
		fmt.Sscanf(r.AudioRate, "%dk", &audioBitrateBPS)

		totalBandwidth := (videoBitrateBPS + audioBitrateBPS) * 1000

		playlistName := fmt.Sprintf("%dp.m3u8", r.Height)
		contentBuilder.WriteString(fmt.Sprintf("#EXT-X-STREAM-INF:BANDWIDTH=%d,RESOLUTION=%dx%d,CODECS=\"avc1.640028,mp4a.40.2\",AUDIO=\"audio\"\n", totalBandwidth, r.Width, r.Height))
		contentBuilder.WriteString(playlistName + "\n")
	}

	return os.WriteFile(masterPlaylistPath, []byte(contentBuilder.String()), 0644)
}
