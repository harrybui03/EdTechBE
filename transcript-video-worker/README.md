# Transcription Worker (Python)

Worker tiêu thụ message RabbitMQ từ cùng exchange với transcode worker (`transcoding_exchange`), polling job status cho đến khi transcode hoàn thành, sau đó tìm audio file và tạo transcript.

## Luồng hoạt động

1. Nhận message từ RabbitMQ (cùng message với transcode worker)
2. Polling job status trong database cho đến khi COMPLETED
3. Tìm audio file trong thư mục entityId (ưu tiên audio.m3u8 hoặc audio file riêng)
4. Extract audio từ HLS nếu cần
5. Transcribe bằng AssemblyAI
6. Upload transcript.json lên MinIO

## Chạy local

1. Tạo và chỉnh `config.local.yaml` theo môi trường của bạn
2. Cài dependency:

```bash
python -m venv .venv && . .venv/Scripts/activate
pip install -r requirements.txt
```

3. Đảm bảo có ffmpeg để extract audio từ HLS:

```bash
# Windows (với Chocolatey)
choco install ffmpeg

# macOS
brew install ffmpeg

# Linux
sudo apt-get install ffmpeg
```

4. Chạy worker:

```bash
python main.py
```

## Message format

```json
{
  "jobId": "UUID",
  "objectPath": "lessons/{lessonId}/videos/{timestamp}-{filename}"
}
```

Kết quả: `transcript.json` được lưu tại cùng thư mục với video trong MinIO.

## Tính năng Translation

Worker tự động dịch transcript sang tiếng Anh bằng AssemblyAI Speech Understanding API. Transcript cuối cùng sẽ luôn là tiếng Anh nếu có translation thành công.

