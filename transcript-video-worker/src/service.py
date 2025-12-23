import io
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import assemblyai as aai
import backoff
import requests
from minio import Minio
from minio.error import S3Error

# Import timeout exceptions từ các HTTP libraries
try:
    from httpx import TimeoutException as HttpxTimeoutException, ReadTimeout as HttpxReadTimeout
except ImportError:
    HttpxTimeoutException = Exception
    HttpxReadTimeout = Exception

try:
    from requests.exceptions import Timeout as RequestsTimeout
except ImportError:
    RequestsTimeout = Exception

from .dto import TranscriptionMessage
from .repository import JobRepository


class TranscriptionService:
    # Constants
    POLL_LOG_INTERVAL_JOB = 12  # Log every 12 polls (1 minute at 5s intervals)
    POLL_LOG_INTERVAL_TRANSCRIPTION = 10  # Log every 10 polls
    POLL_LOG_INTERVAL_TRANSLATION = 10  # Log every 10 polls for translation
    
    # Valid AssemblyAI language codes (as of 2025)
    VALID_LANGUAGE_CODES = {
        'en', 'es', 'fr', 'de', 'it', 'pt', 'nl', 'hi', 'ja', 'zh', 
        'fi', 'ko', 'pl', 'ru', 'tr', 'uk', 'vi', 'ar', 'da', 'el',
        'id', 'ms', 'no', 'ro', 'sv', 'th', 'cs', 'hu', 'sk', 'bg'
    }
    
    def __init__(self, cfg: dict, job_repo: JobRepository) -> None:
        self._bucket = cfg["minio"]["bucket"]
        self._minio = Minio(
            cfg["minio"]["url"],
            access_key=cfg["minio"]["access_id"],
            secret_key=cfg["minio"]["secret_access_key"],
            secure=False,
        )
        self._logger = logging.getLogger("transcription.service")
        self._job_repo = job_repo
        
        api_key = cfg.get("assemblyai", {}).get("api_key")
        if not api_key:
            raise ValueError("AssemblyAI API key is required. Set it in config.yaml")
        aai.settings.api_key = api_key
        
        # Check FFmpeg availability
        if not shutil.which("ffmpeg"):
            self._logger.warning("⚠️  FFmpeg not found in PATH. HLS audio extraction will fail.")
        
        # Đảm bảo thư mục transcripts tồn tại
        self._transcripts_dir = Path("transcripts")
        self._transcripts_dir.mkdir(exist_ok=True)
        
        # Timeout settings cho AssemblyAI
        self._transcribe_timeout = int(cfg.get("assemblyai", {}).get("timeout", "300"))  # 5 phút mặc định
        self._poll_timeout = int(cfg.get("assemblyai", {}).get("poll_timeout", "1800"))  # 30 phút cho polling
        self._poll_interval = int(cfg.get("assemblyai", {}).get("poll_interval", "5"))  # 5 giây giữa các lần poll
        
        # Job polling settings
        self._job_poll_interval = int(cfg.get("job_polling", {}).get("interval", "5"))  # 5 giây mặc định
        self._job_poll_timeout = int(cfg.get("job_polling", {}).get("timeout", "3600"))  # 1 giờ mặc định

    def _poll_job_status(self, job_id: str) -> dict:
        """Polling job status cho đến khi COMPLETED hoặc FAILED"""
        start_time = time.time()
        poll_count = 0
        
        self._logger.info(f"⏳ Polling job status | jobId={job_id[:8]}...")
        
        while True:
            elapsed = time.time() - start_time
            if elapsed > self._job_poll_timeout:
                raise TimeoutError(
                    f"Job polling timeout after {elapsed:.0f}s. "
                    f"Job ID: {job_id}"
                )
            
            job = self._job_repo.find_job_by_id(job_id)
            if not job:
                raise ValueError(f"Job not found: {job_id}")
            
            status = job.get("status")
            entity_id = job.get("entity_id")
            
            poll_count += 1
            if poll_count % self.POLL_LOG_INTERVAL_JOB == 0:  # Log mỗi 1 phút (12 * 5s)
                self._logger.info(
                    f"   ⏳ Job status: {status} | "
                    f"elapsed: {elapsed:.0f}s | poll: {poll_count}"
                )
            
            if status == "COMPLETED":
                self._logger.info(f"✅ Job completed | jobId={job_id[:8]}... | entityId={entity_id}")
                return job
            elif status == "FAILED":
                raise Exception(f"Job failed: {job_id}")
            
            time.sleep(self._job_poll_interval)
    
    def _find_audio_file(self, entity_id: str, video_dir: str) -> Optional[str]:
        """Tìm audio file trong thư mục entityId. Ưu tiên audio.m3u8 hoặc audio file riêng"""
        # Thư mục chứa video: lessons/{entityId}/videos/...
        # Tìm trong thư mục đó
        prefix = f"lessons/{entity_id}/videos/"
        
        try:
            objects = self._minio.list_objects(self._bucket, prefix=prefix, recursive=True)
            
            # Ưu tiên tìm audio.m3u8 (audio track từ HLS)
            audio_m3u8 = None
            audio_files = []
            
            for obj in objects:
                object_name = obj.object_name
                if object_name.endswith("audio.m3u8"):
                    audio_m3u8 = object_name
                elif any(object_name.endswith(ext) for ext in [".m4a", ".mp3", ".wav", ".aac", ".ogg"]):
                    audio_files.append(object_name)
            
            # Ưu tiên audio.m3u8, sau đó là audio file riêng
            if audio_m3u8:
                self._logger.info(f"   🎵 Found audio.m3u8: {audio_m3u8}")
                return audio_m3u8
            elif audio_files:
                self._logger.info(f"   🎵 Found audio file: {audio_files[0]}")
                return audio_files[0]
            else:
                # Nếu không có audio file riêng, sẽ extract từ video HLS
                # Tìm master.m3u8 để extract audio
                master_m3u8 = f"{prefix}master.m3u8"
                try:
                    self._minio.stat_object(self._bucket, master_m3u8)
                    self._logger.info(f"   🎬 Found master.m3u8, will extract audio from HLS")
                    return master_m3u8
                except S3Error:
                    self._logger.warning(f"   ⚠️  No audio file found in {prefix}")
                    return None
                    
        except Exception as e:
            self._logger.error(f"❌ Failed to list objects in {prefix}: {e}")
            return None
    
    def _extract_audio_from_hls(self, hls_path: str, output_path: Path) -> Path:
        """Extract audio từ HLS playlist thành file audio riêng"""
        # Download HLS playlist và extract audio
        temp_dir = output_path.parent
        audio_output = temp_dir / "extracted_audio.m4a"
        hls_local = temp_dir / "audio.m3u8"
        
        self._logger.info(f"   🎬 Extracting audio from HLS: {hls_path}")
        
        segment_files = []
        concat_file = temp_dir / "concat_list.txt"
        
        try:
            # Download audio.m3u8 playlist
            self._minio.fget_object(self._bucket, hls_path, str(hls_local))
            
            # Đọc playlist và download các segments
            with open(hls_local, "r") as f:
                playlist_content = f.read()
            
            # Download các audio segments
            lines = playlist_content.split("\n")
            for line in lines:
                line = line.strip()
                if line and not line.startswith("#") and line.endswith(".ts"):
                    segment_path = os.path.join(os.path.dirname(hls_path), line)
                    segment_local = temp_dir / line
                    try:
                        self._minio.fget_object(self._bucket, segment_path, str(segment_local))
                        segment_files.append(str(segment_local))
                    except Exception as e:
                        self._logger.warning(f"   ⚠️  Failed to download segment {line}: {e}")
            
            if not segment_files:
                raise Exception("No audio segments found in playlist")
            
            self._logger.info(f"   📦 Downloaded {len(segment_files)} segments")
            
            # Sử dụng ffmpeg để concat và convert segments thành audio file
            # Tạo file list cho ffmpeg concat
            with open(concat_file, "w") as f:
                for seg in segment_files:
                    f.write(f"file '{seg}'\n")
            
            # Extract audio bằng ffmpeg
            cmd = [
                "ffmpeg",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_file),
                "-vn",  # No video
                "-acodec", "aac",  # Convert to AAC
                "-b:a", "192k",  # Audio bitrate
                "-y",  # Overwrite output
                str(audio_output)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                raise Exception(
                    f"FFmpeg failed with return code {result.returncode}\n"
                    f"STDERR: {result.stderr}\nSTDOUT: {result.stdout}"
                )
            
            if audio_output.exists() and audio_output.stat().st_size > 0:
                self._logger.info(f"   ✅ Audio extracted: {audio_output} ({audio_output.stat().st_size} bytes)")
                return audio_output
            else:
                raise Exception("Audio extraction failed: output file not found or empty")
                
        except Exception as e:
            self._logger.error(f"❌ Failed to extract audio from HLS: {e}")
            raise
        finally:
            # Clean up segment files and concat list
            for seg in segment_files:
                try:
                    Path(seg).unlink(missing_ok=True)
                except Exception as e:
                    self._logger.debug(f"Failed to cleanup segment {seg}: {e}")
            try:
                concat_file.unlink(missing_ok=True)
            except Exception as e:
                self._logger.debug(f"Failed to cleanup concat file: {e}")

    def process(self, msg: TranscriptionMessage) -> None:
        """Process transcription message: polling job, find audio, transcribe, upload transcript"""
        job_id_short = msg.jobId[:8] if len(msg.jobId) > 8 else msg.jobId
        self._logger.info(f"🔄 Processing transcription | jobId={job_id_short}... | objectPath={msg.objectPath}")
        
        # 1. Polling job status cho đến khi COMPLETED
        job = self._poll_job_status(msg.jobId)
        entity_id = str(job.get("entity_id"))
        
        # 2. Extract entityId từ objectPath để tìm audio file
        # objectPath format: lessons/{entityId}/videos/{timestamp}-{filename}
        video_dir = os.path.dirname(msg.objectPath)
        
        # 3. Tìm audio file trong thư mục entityId
        audio_path = self._find_audio_file(entity_id, video_dir)
        if not audio_path:
            raise Exception(f"No audio file found for entityId: {entity_id}")
        
        # 4. Download và extract audio nếu cần
        temp_dir_path = None
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir_path = temp_dir
                audio_file = None
                
                if audio_path.endswith(".m3u8"):
                    # Extract audio từ HLS
                    audio_file = self._extract_audio_from_hls(audio_path, Path(temp_dir) / "audio.m4a")
                else:
                    # Download audio file trực tiếp
                    audio_file = Path(temp_dir) / Path(audio_path).name
                    self._logger.debug(f"   📥 Downloading audio | bucket={self._bucket} | object={audio_path}")
                    try:
                        self._minio.fget_object(self._bucket, audio_path, str(audio_file))
                        self._logger.debug(f"   ✅ Audio downloaded | size={audio_file.stat().st_size} bytes")
                    except S3Error as e:
                        self._logger.error(f"❌ Failed to download audio: {e}")
                        raise
                
                # Validate audio file
                if not audio_file.exists() or audio_file.stat().st_size == 0:
                    raise Exception(f"Audio file is missing or empty: {audio_file}")
                
                # Transcribe bằng AssemblyAI
                self._logger.info(f"🎤 Transcribing audio with AssemblyAI...")
                
                try:
                    # Upload audio file lên AssemblyAI với retry
                    transcript = aai.Transcriber()
                    
                    # Configure transcription
                    # Luôn enable language_detection để detect language tự động
                    # Kích hoạt translation sang tiếng Anh
                    translation_request = aai.TranslationRequest(
                        target_languages=["en"]  # Target là tiếng Anh
                    )
                    speech_understanding = aai.SpeechUnderstandingRequest(
                        request=aai.SpeechUnderstandingFeatureRequests(
                            translation=translation_request
                        )
                    )
                    
                    config = aai.TranscriptionConfig(
                        language_detection=True,
                        speech_model=aai.SpeechModel.universal,  # Model cân bằng giữa độ chính xác và chi phí
                        speech_understanding=speech_understanding
                    )
                    
                    if msg.language:
                        # Validate language code
                        if msg.language in self.VALID_LANGUAGE_CODES:
                            config.language_code = msg.language
                    
                    # Transcribe với retry
                    @backoff.on_exception(
                        backoff.expo,
                        (TimeoutError, RequestsTimeout, HttpxTimeoutException, HttpxReadTimeout, Exception),
                        max_tries=3,
                        max_time=self._transcribe_timeout,
                        jitter=backoff.full_jitter,
                    )
                    def _transcribe_audio():
                        return transcript.transcribe(str(audio_file), config=config)
                    
                    result = _transcribe_audio()
                    
                    # Poll cho đến khi complete với timeout
                    start_time = time.time()
                    poll_count = 0
                    while result.status == aai.TranscriptStatus.queued or result.status == aai.TranscriptStatus.processing:
                        elapsed = time.time() - start_time
                        if elapsed > self._poll_timeout:
                            raise TimeoutError(
                                f"Transcription polling timeout after {elapsed:.0f}s. "
                                f"Transcript ID: {result.id}, Status: {result.status}"
                            )
                        
                        poll_count += 1
                        if poll_count % self.POLL_LOG_INTERVAL_TRANSCRIPTION == 0:  # Log mỗi 10 lần poll
                            self._logger.info(
                                f"   ⏳ Transcription status: {result.status} | "
                                f"elapsed: {elapsed:.0f}s | poll: {poll_count}"
                            )
                        else:
                            self._logger.debug(f"   ⏳ Transcription status: {result.status}")
                        
                        time.sleep(self._poll_interval)
                        
                        # Get transcript với retry
                        @backoff.on_exception(
                            backoff.expo,
                            (TimeoutError, RequestsTimeout, HttpxTimeoutException, Exception),
                            max_tries=3,
                            max_time=30,
                            jitter=backoff.full_jitter,
                        )
                        def _get_transcript():
                            transcript_result = aai.Transcript.get_by_id(result.id)
                            
                            # Log chi tiết response từ API
                            status = transcript_result.status
                            language = getattr(transcript_result, 'language_code', 'unknown')
                            has_translated_texts = hasattr(transcript_result, 'translated_texts') and transcript_result.translated_texts
                            has_speech_understanding = hasattr(transcript_result, 'speech_understanding') and transcript_result.speech_understanding
                            
                            self._logger.debug(
                                f"   📊 Transcript response | id={result.id[:8]}... | "
                                f"status={status} | language={language} | "
                                f"has_translated_texts={bool(has_translated_texts)} | "
                                f"has_speech_understanding={bool(has_speech_understanding)}"
                            )
                            
                            # Log toàn bộ response dưới dạng dict nếu có thể
                            try:
                                # Thử convert transcript object thành dict để log
                                response_dict = {}
                                if hasattr(transcript_result, '__dict__'):
                                    response_dict = transcript_result.__dict__.copy()
                                else:
                                    # Nếu không có __dict__, thử các attributes quan trọng
                                    response_dict = {
                                        'id': getattr(transcript_result, 'id', None),
                                        'status': status,
                                        'language_code': language,
                                        'text': getattr(transcript_result, 'text', None)[:100] + "..." if hasattr(transcript_result, 'text') and transcript_result.text else None,
                                        'has_translated_texts': bool(has_translated_texts),
                                        'has_speech_understanding': bool(has_speech_understanding),
                                    }
                                
                                # Log JSON response (chỉ các fields quan trọng để không quá dài)
                                self._logger.debug(f"   📋 Response data: {json.dumps(response_dict, default=str, ensure_ascii=False)[:500]}")
                            except Exception as e:
                                self._logger.debug(f"   ⚠️  Could not serialize response: {e}")
                            
                            # Log chi tiết hơn nếu có translated_texts hoặc speech_understanding
                            if has_translated_texts:
                                if isinstance(transcript_result.translated_texts, dict):
                                    translated_langs = list(transcript_result.translated_texts.keys())
                                    self._logger.debug(f"      🌐 Translated languages: {translated_langs}")
                                    # Log một phần translated text nếu có
                                    if "en" in transcript_result.translated_texts:
                                        en_text = transcript_result.translated_texts["en"]
                                        if en_text:
                                            preview = str(en_text)[:200] + "..." if len(str(en_text)) > 200 else str(en_text)
                                            self._logger.debug(f"      📝 English translation preview: {preview}")
                                else:
                                    self._logger.debug(f"      🌐 Translated texts: available (object)")
                            
                            if has_speech_understanding:
                                if isinstance(transcript_result.speech_understanding, dict):
                                    su_response = transcript_result.speech_understanding.get('response', {})
                                    translation_status = su_response.get('translation', {}).get('status', 'unknown')
                                    self._logger.debug(f"      🎯 Speech understanding: {json.dumps(transcript_result.speech_understanding, default=str, ensure_ascii=False)[:300]}")
                                else:
                                    su_response = getattr(transcript_result.speech_understanding, 'response', None)
                                    translation = getattr(su_response, 'translation', None) if su_response else None
                                    translation_status = getattr(translation, 'status', 'unknown') if translation else 'unknown'
                                    self._logger.debug(f"      🎯 Speech understanding translation status: {translation_status}")
                            
                            return transcript_result
                        
                        result = _get_transcript()
                    
                    if result.status == aai.TranscriptStatus.error:
                        error_msg = result.error if hasattr(result, 'error') else "Unknown error"
                        raise Exception(f"AssemblyAI transcription failed: {error_msg}")
                    
                    # Lấy kết quả
                    detected_language = result.language_code if hasattr(result, 'language_code') else "unknown"
                    duration = result.audio_duration / 1000.0 if result.audio_duration else 0  # Convert ms to seconds
                    original_text = result.text.strip() if result.text else ""
                    
                    # Lấy kết quả dịch từ translated_texts["en"] (quan trọng)
                    translated_text = None
                    if hasattr(result, 'translated_texts') and result.translated_texts:
                        # translated_texts có thể là dict hoặc object, xử lý cả 2 trường hợp
                        if isinstance(result.translated_texts, dict):
                            translated_text = result.translated_texts.get("en", "").strip() or None
                        elif hasattr(result.translated_texts, 'en'):
                            translated_text = str(getattr(result.translated_texts, 'en', "")).strip() or None
                        else:
                            translated_text = None
                        
                        if translated_text:
                            self._logger.info(f"✅ Translation included in transcription | language={detected_language}")
                        else:
                            self._logger.warning(f"⚠️  Translation config enabled but no English translation found | language={detected_language}")
                    
                    # Nếu không có translation và language != "en", cần translate
                    # Kiểm tra lại detected_language từ kết quả (có thể khác với msg.language)
                    if not translated_text and detected_language and detected_language != "en":
                        self._logger.info(f"🌐 Adding translation to existing transcript...")
                        try:
                            # Method 2: Add translation to existing transcript using Speech Understanding API
                            # Gửi transcript_id đến Speech Understanding API
                            base_url = "https://llm-gateway.assemblyai.com/v1/understanding"
                            headers = {
                                "Authorization": aai.settings.api_key,
                                "Content-Type": "application/json"
                            }
                            
                            data = {
                                "transcript_id": result.id,
                                "speech_understanding": {
                                    "request": {
                                        "translation": {
                                            "target_languages": ["en"],
                                            "formal": False
                                        }
                                    }
                                }
                            }
                            
                            @backoff.on_exception(
                                backoff.expo,
                                (TimeoutError, RequestsTimeout, HttpxTimeoutException, Exception),
                                max_tries=3,
                                max_time=300,
                                jitter=backoff.full_jitter,
                            )
                            def _request_translation():
                                response = requests.post(base_url, headers=headers, json=data, timeout=30)
                                response.raise_for_status()
                                return response.json()
                            
                            translation_response = _request_translation()
                            
                            # Poll transcript để lấy kết quả translation
                            translate_start_time = time.time()
                            translate_poll_count = 0
                            while True:
                                translate_elapsed = time.time() - translate_start_time
                                if translate_elapsed > self._poll_timeout:
                                    raise TimeoutError(f"Translation polling timeout after {translate_elapsed:.0f}s")
                                
                                # Get updated transcript với translation
                                # Tạo Transcriber instance mới để get transcript
                                @backoff.on_exception(
                                    backoff.expo,
                                    (TimeoutError, RequestsTimeout, HttpxTimeoutException, Exception),
                                    max_tries=3,
                                    max_time=30,
                                    jitter=backoff.full_jitter,
                                )
                                def _get_translated_transcript():
                                    transcript_result = aai.Transcript.get_by_id(result.id)
                                    
                                    # Log chi tiết response từ API
                                    status = transcript_result.status
                                    language = getattr(transcript_result, 'language_code', 'unknown')
                                    has_translated_texts = hasattr(transcript_result, 'translated_texts') and transcript_result.translated_texts
                                    has_speech_understanding = hasattr(transcript_result, 'speech_understanding') and transcript_result.speech_understanding
                                    
                                    self._logger.debug(
                                        f"   📊 Transcript response (translation) | id={result.id[:8]}... | "
                                        f"status={status} | language={language} | "
                                        f"has_translated_texts={bool(has_translated_texts)} | "
                                        f"has_speech_understanding={bool(has_speech_understanding)}"
                                    )
                                    
                                    # Log toàn bộ response dưới dạng dict nếu có thể
                                    try:
                                        # Thử convert transcript object thành dict để log
                                        response_dict = {}
                                        if hasattr(transcript_result, '__dict__'):
                                            response_dict = transcript_result.__dict__.copy()
                                        else:
                                            # Nếu không có __dict__, thử các attributes quan trọng
                                            response_dict = {
                                                'id': getattr(transcript_result, 'id', None),
                                                'status': status,
                                                'language_code': language,
                                                'text': getattr(transcript_result, 'text', None)[:100] + "..." if hasattr(transcript_result, 'text') and transcript_result.text else None,
                                                'has_translated_texts': bool(has_translated_texts),
                                                'has_speech_understanding': bool(has_speech_understanding),
                                            }
                                        
                                        # Log JSON response (chỉ các fields quan trọng để không quá dài)
                                        self._logger.debug(f"   📋 Response data: {json.dumps(response_dict, default=str, ensure_ascii=False)[:500]}")
                                    except Exception as e:
                                        self._logger.debug(f"   ⚠️  Could not serialize response: {e}")
                                    
                                    # Log chi tiết hơn nếu có translated_texts hoặc speech_understanding
                                    if has_translated_texts:
                                        if isinstance(transcript_result.translated_texts, dict):
                                            translated_langs = list(transcript_result.translated_texts.keys())
                                            self._logger.debug(f"      🌐 Translated languages: {translated_langs}")
                                            # Log một phần translated text nếu có
                                            if "en" in transcript_result.translated_texts:
                                                en_text = transcript_result.translated_texts["en"]
                                                if en_text:
                                                    preview = str(en_text)[:200] + "..." if len(str(en_text)) > 200 else str(en_text)
                                                    self._logger.debug(f"      📝 English translation preview: {preview}")
                                        else:
                                            self._logger.debug(f"      🌐 Translated texts: available (object)")
                                    
                                    if has_speech_understanding:
                                        if isinstance(transcript_result.speech_understanding, dict):
                                            su_response = transcript_result.speech_understanding.get('response', {})
                                            translation_status = su_response.get('translation', {}).get('status', 'unknown')
                                            self._logger.debug(f"      🎯 Speech understanding: {json.dumps(transcript_result.speech_understanding, default=str, ensure_ascii=False)[:300]}")
                                        else:
                                            su_response = getattr(transcript_result.speech_understanding, 'response', None)
                                            translation = getattr(su_response, 'translation', None) if su_response else None
                                            translation_status = getattr(translation, 'status', 'unknown') if translation else 'unknown'
                                            self._logger.debug(f"      🎯 Speech understanding translation status: {translation_status}")
                                    
                                    return transcript_result
                                
                                updated_result = _get_translated_transcript()
                                
                                # Kiểm tra transcript status - nếu completed nhưng không có speech_understanding sau một thời gian, dừng
                                transcript_status = updated_result.status
                                has_speech_understanding = hasattr(updated_result, 'speech_understanding') and updated_result.speech_understanding
                                
                                if transcript_status == aai.TranscriptStatus.completed:
                                    # Nếu transcript đã completed nhưng không có speech_understanding sau 60 giây, 
                                    # có nghĩa là translation request không được xử lý hoặc đã fail
                                    if translate_elapsed > 60 and not has_speech_understanding:
                                        self._logger.warning(
                                            f"⚠️  Transcript completed but no translation response after {translate_elapsed:.0f}s, "
                                            f"stopping translation polling"
                                        )
                                        break
                                
                                # Kiểm tra xem translation đã hoàn thành chưa
                                if has_speech_understanding:
                                    speech_understanding = updated_result.speech_understanding
                                    # speech_understanding có thể là dict hoặc object
                                    translation_status = None
                                    if isinstance(speech_understanding, dict):
                                        response_data = speech_understanding.get('response', {})
                                        translation_status = response_data.get('translation', {}).get('status')
                                    else:
                                        # Nếu là object, thử truy cập attribute
                                        response = getattr(speech_understanding, 'response', None)
                                        translation = getattr(response, 'translation', None) if response else None
                                        translation_status = getattr(translation, 'status', None) if translation else None
                                    
                                    # Handle terminal states
                                    if translation_status in ('failed', 'error'):
                                        self._logger.warning(f"⚠️  Translation failed with status: {translation_status}")
                                        break
                                    elif translation_status == 'success':
                                        # Lấy translated_texts
                                        if hasattr(updated_result, 'translated_texts') and updated_result.translated_texts:
                                            if isinstance(updated_result.translated_texts, dict):
                                                translated_text = updated_result.translated_texts.get("en", "").strip() or None
                                            else:
                                                translated_text = getattr(updated_result.translated_texts, 'en', None)
                                                if translated_text:
                                                    translated_text = str(translated_text).strip() or None
                                            
                                            if translated_text:
                                                self._logger.info(f"✅ Translation completed via Speech Understanding API")
                                                break
                                
                                translate_poll_count += 1
                                if translate_poll_count % self.POLL_LOG_INTERVAL_TRANSLATION == 0:
                                    self._logger.debug(f"   ⏳ Waiting for translation | elapsed: {translate_elapsed:.0f}s")
                                
                                time.sleep(self._poll_interval)
                                
                        except Exception as e:
                            self._logger.warning(f"⚠️  Translation failed, keeping original: {e}")
                    
                    self._logger.info(
                        f"✅ Transcription complete | language={detected_language} | "
                        f"duration={duration:.2f}s | has_translation={bool(translated_text)}"
                    )
                    
                except (TimeoutError, RequestsTimeout, HttpxTimeoutException, HttpxReadTimeout) as e:
                    self._logger.error(
                        f"❌ Transcription timeout: {e} | "
                        f"audioPath={audio_path} | "
                        f"fileSize={audio_file.stat().st_size if audio_file and audio_file.exists() else 'unknown'} bytes"
                    )
                    raise
                except Exception as e:
                    error_type = type(e).__name__
                    self._logger.error(
                        f"❌ Transcription failed: {error_type}: {e} | "
                        f"audioPath={audio_path}",
                        exc_info=True
                    )
                    raise
                
                # Text sẽ luôn là tiếng Anh nếu có translation, nếu không thì dùng original text
                final_text = translated_text if translated_text else original_text
                
                payload = {
                    "lessonId": entity_id,
                    "jobId": msg.jobId,
                    "audioPath": audio_path,
                    "model": "assemblyai",
                    "language": detected_language,
                    "createdAt": datetime.now(timezone.utc).isoformat(),
                    "version": 1,
                    "duration": duration,
                    "text": final_text,  # Luôn là tiếng Anh nếu có translation
                }
                
                # Upload transcript lên MinIO
                data_bytes = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
                data_stream = io.BytesIO(data_bytes)
                
                # Lưu transcript cùng thư mục với video
                dir_prefix = video_dir.replace("\\", "/")
                object_name = f"{dir_prefix}/transcript.json" if dir_prefix else "transcript.json"
                
                self._logger.debug(f"   📤 Uploading transcript | bucket={self._bucket} | object={object_name}")
                try:
                    self._minio.put_object(
                        self._bucket,
                        object_name,
                        data=data_stream,
                        length=len(data_bytes),
                        content_type="application/json",
                    )
                    self._logger.info(
                        f"✅ Transcript uploaded | jobId={job_id_short}... | "
                        f"object={object_name} | duration={duration:.2f}s"
                    )
                except S3Error as e:
                    self._logger.error(
                        f"❌ MinIO upload failed | jobId={job_id_short}... | "
                        f"bucket={self._bucket} | object={object_name} | error={e}"
                    )
                    raise
                except Exception as e:
                    self._logger.error(
                        f"❌ Unexpected error during upload | jobId={job_id_short}... | error={e}",
                        exc_info=True
                    )
                    raise
                
                # Lưu transcript vào thư mục transcripts với tên file là entityId
                try:
                    transcript_file = self._transcripts_dir / f"{entity_id}.json"
                    with open(transcript_file, "w", encoding="utf-8") as f:
                        json.dump(payload, f, ensure_ascii=False, indent=2)
                    self._logger.info(
                        f"💾 Transcript saved locally | entityId={entity_id[:8]}... | "
                        f"file={transcript_file}"
                    )
                except Exception as e:
                    self._logger.warning(
                        f"⚠️  Failed to save transcript locally | entityId={entity_id[:8]}... | error={e}"
                    )
                    # Không raise exception vì đây là backup, không phải critical
        
        except Exception as e:
            # Explicit cleanup on critical failure (in case tempfile context manager fails)
            if temp_dir_path and Path(temp_dir_path).exists():
                try:
                    shutil.rmtree(temp_dir_path, ignore_errors=True)
                    self._logger.debug(f"   🧹 Cleaned up temp directory after failure: {temp_dir_path}")
                except Exception as cleanup_error:
                    self._logger.debug(f"   ⚠️  Failed to cleanup temp directory: {cleanup_error}")
            raise