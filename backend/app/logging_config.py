import logging
import logging.handlers
from pathlib import Path
import os
from typing import Optional


# ==================== Logging Constants ====================
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_FILE = LOG_DIR / "ldsc.log"
LOG_FORMAT = "%(asctime)s - %(levelname)-8s - %(name)s:%(funcName)s:%(lineno)d - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
MAX_BYTES = 10 * 1024 * 1024  # 10MB
BACKUP_COUNT = 5


def create_log_directory() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def get_console_handler(level: int) -> logging.StreamHandler:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    console_handler.setFormatter(formatter)
    
    return console_handler


def get_file_handler(level: int) -> logging.handlers.RotatingFileHandler:
    create_log_directory()
    
    file_handler = logging.handlers.RotatingFileHandler(
        filename=LOG_FILE,
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8"
    )
    file_handler.setLevel(level)
    
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    file_handler.setFormatter(formatter)
    
    return file_handler


def setup_logging(log_level: str = "INFO") -> None:

    # Validate and convert log level string
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler
    console_handler = get_console_handler(numeric_level)
    root_logger.addHandler(console_handler)
    
    # Add file handler
    file_handler = get_file_handler(numeric_level)
    root_logger.addHandler(file_handler)
    
    # Set logging for specific libraries
    logging.getLogger("uvicorn").setLevel(numeric_level)
    logging.getLogger("uvicorn.access").setLevel(numeric_level)
    logging.getLogger("fastapi").setLevel(numeric_level)
    logging.getLogger("torch").setLevel(logging.WARNING)  # Suppress PyTorch info logs


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


# Initialize logging on module import
if not logging.getLogger().handlers:
    create_log_directory()
    setup_logging("INFO")
