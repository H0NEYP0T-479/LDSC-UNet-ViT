import logging
from app.logging_config import get_logger


def log_inference(logger: logging.Logger, image_id: str, stage: str, message: str) -> None:
    """Log structured inference step."""
    logger.info(f"[{image_id}] [{stage}] {message}")


def log_error(logger: logging.Logger, image_id: str, error: Exception) -> None:
    """Log structured error with context."""
    logger.error(f"[{image_id}] ERROR: {type(error).__name__}: {str(error)}")