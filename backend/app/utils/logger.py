import logging
import traceback
from typing import Optional
from app.logging_config import get_logger as _get_logger


def get_logger(name: str) -> logging.Logger:
    return _get_logger(name)


def log_inference(
    logger: logging.Logger,
    image_id: str,
    stage: str,
    message: str
) -> None:
    formatted_message = f"[{image_id} | {stage}] {message}"
    logger.info(formatted_message)


def log_error(
    logger: logging.Logger,
    image_id: str,
    error: Exception,
    stage: Optional[str] = None
) -> None:
    if stage:
        formatted_message = f"[{image_id} | {stage}] Error: {type(error).__name__}: {str(error)}"
    else:
        formatted_message = f"[{image_id}] Error: {type(error).__name__}: {str(error)}"
    
    logger.error(formatted_message, exc_info=True)
