import logging
from typing import Optional
import os


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> logging.Logger:
    """Set up a logger with console and file handlers.

    Args:
        name: Name of the logger (typically __name__)
        log_file: Path to log file. If None, only console handler is added
        console_level: Logging level for console output
        file_level: Logging level for file output

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(min(console_level, file_level))  # Set to lowest level of the two

    # Create formatters
    console_format = logging.Formatter(
        "[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S"
    )
    file_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create and configure console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # Create and configure file handler if log file is specified
    if log_file:
        # Create the directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level)
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger
