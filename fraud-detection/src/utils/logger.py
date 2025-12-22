import logging
from pathlib import Path

def get_logger(log_file_name: str) -> logging.Logger:
    """
    Returns a configured logger instance.

    Parameters:
    - log_file_name: str - the name of the log file (e.g., "my_log.log")

    Returns:
    - logger: logging.Logger instance
    """
    # Fixed logs folder at the project root
    root_dir = Path(__file__).resolve().parent.parent  # assuming src/utils/logger.py
    log_path = root_dir / "logs"
    log_path.mkdir(parents=True, exist_ok=True)

    # Full path to the log file
    log_file = log_path / log_file_name

    # Logger instance
    logger = logging.getLogger(log_file_name)
    logger.setLevel(logging.DEBUG)

    # Avoid adding multiple handlers
    if not logger.handlers:
        # File handler (append mode)
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_formatter = logging.Formatter(
            '%(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)

        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
