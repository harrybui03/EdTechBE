import logging
import signal
import sys

from src.config import load_config
from src.consumer import TranscriptionConsumer
from src.service import TranscriptionService
from src.repository import JobRepository


class ColoredFormatter(logging.Formatter):
    """Formatter với màu sắc cho console output"""
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",   # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        log_color = self.COLORS.get(record.levelname, "")
        record.levelname = f"{log_color}{record.levelname:8s}{self.RESET}"
        return super().format(record)


def setup_logger(env: str) -> None:
    """Setup logger với format đẹp và tắt noise từ thư viện bên thứ 3"""
    # Set level cho root logger
    root_level = logging.DEBUG if env == "develop" else logging.INFO
    root_logger = logging.getLogger()
    root_logger.setLevel(root_level)

    # Tắt DEBUG logs từ các thư viện bên thứ 3
    logging.getLogger("pika").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("minio").setLevel(logging.WARNING)
    logging.getLogger("backoff").setLevel(logging.WARNING)

    # Console handler với màu sắc
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(root_level)
    
    # Format đẹp hơn với màu sắc
    console_format = ColoredFormatter(
        "%(asctime)s | %(levelname)s | %(name)-20s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)

    # File handler (optional, có thể bật nếu cần)
    # file_handler = RotatingFileHandler(
    #     "transcription-worker.log",
    #     maxBytes=10 * 1024 * 1024,  # 10MB
    #     backupCount=5
    # )
    # file_handler.setLevel(logging.INFO)
    # file_format = logging.Formatter(
    #     "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
    #     datefmt="%Y-%m-%d %H:%M:%S"
    # )
    # file_handler.setFormatter(file_format)
    # root_logger.addHandler(file_handler)


def main() -> int:
    cfg = load_config("../config.yaml")
    setup_logger(cfg["app"]["environment"])
    
    logger = logging.getLogger("transcription.main")
    logger.info("=" * 60)
    logger.info("🚀 Starting Transcription Worker")
    logger.info(f"   Environment: {cfg['app']['environment']}")
    logger.info(f"   Workers: {cfg['server']['workers']}")
    logger.info("=" * 60)

    job_repo = JobRepository(cfg)
    service = TranscriptionService(cfg, job_repo)
    consumer = TranscriptionConsumer(cfg, service)

    def handle_sigterm(signum, frame):
        logger.info("🛑 Received shutdown signal, shutting down gracefully...")
        consumer.stop()

    signal.signal(signal.SIGINT, handle_sigterm)
    signal.signal(signal.SIGTERM, handle_sigterm)

    try:
        consumer.start()
    except KeyboardInterrupt:
        logger.info("🛑 Interrupted by user")
    except Exception as e:
        logger.exception(f"💥 Fatal error: {e}")
        return 1
    finally:
        consumer.stop()

    logger.info("✅ Transcription worker stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())


