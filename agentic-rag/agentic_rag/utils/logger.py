"""
Logging utility for Agentic RAG.
Provides both console logging and trace capture for API responses.
"""

import logging
import sys
from io import StringIO
from typing import Optional


class TraceCaptureHandler(logging.Handler):
    """
    Custom logging handler that captures log messages to a buffer
    while also outputting to console.
    """
    
    def __init__(self, buffer: StringIO):
        super().__init__()
        self.buffer = buffer
        self.console_handler = logging.StreamHandler(sys.stdout)
        self.console_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
    
    def emit(self, record):
        """Emit log record to both buffer and console"""
        try:
            msg = self.format(record)
            # Write to buffer for trace capture
            self.buffer.write(msg + '\n')
            # Also write to console
            self.console_handler.emit(record)
        except Exception:
            self.handleError(record)


def setup_logger(name: str = "agentic_rag", level: int = logging.INFO) -> logging.Logger:
    """
    Setup logger with console output.
    
    Args:
        name: Logger name
        level: Logging level (default: INFO)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Console handler with formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get or create a logger instance.
    
    Args:
        name: Optional logger name (default: "agentic_rag")
        
    Returns:
        Logger instance
    """
    if name is None:
        name = "agentic_rag"
    return logging.getLogger(name)


# Create default logger instance
default_logger = setup_logger()

