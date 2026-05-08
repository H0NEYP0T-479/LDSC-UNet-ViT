import logging
import logging.handlers
import os


def setup_logging(log_level: str = "INFO") -> None:
    """Set up structured logging with console and file handlers."""
    os.makedirs("logs", exist_ok=True)

    fmt = "%(asctime)s | %(levelname)s | %(module)s | %(message)s"
    formatter = logging.Formatter(fmt)

    console = logging.StreamHandler()
    console.setFormatter(formatter)

    file_handler = logging.handlers.RotatingFileHandler(
        "logs/ldsc.log", maxBytes=10 * 1024 * 1024, backupCount=5
    )
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    root.addHandler(console)
    root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger."""
    return logging.getLogger(name)