import logging
import os
from pathlib import Path

def get_logger(log_file_name: str) -> logging.Logger:
    """
    Returns a configured logger instance with logs at project root.
    
    Parameters:
    - log_file_name: str - the name of the log file (e.g., "my_log.log")
    
    Returns:
    - logger: logging.Logger instance
    """
    # Method 1: Go up from current file location (adjust based on your structure)
    # Assuming logger.py is in src/utils/
    current_file = Path(__file__).resolve()
    
    # Navigate to project root - adjust the number of .parent based on your structure
    # src/utils/logger.py -> go up 2 levels to reach project root
    project_root = current_file.parent.parent.parent  # Adjust this!
    
    # OR Method 2: Use environment variable or fixed path
    # project_root = Path("/home/ermias/Desktop/fraud_task/Improved_detection_of_fraud_cases_for_e-commerce_and_bank_transactions/fraud-detection")
    
    # OR Method 3: Use current working directory (where script is run from)
    # project_root = Path.cwd()
    
    log_path = project_root / "logs"
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Full path to the log file
    log_file = log_path / log_file_name
    
    print(f"Log file will be created at: {log_file}")  # Debug print
    
    # Rest of your logger configuration...
    logger_name = Path(log_file_name).stem
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    
    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(log_file, mode="a", encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_formatter = logging.Formatter(
            '%(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger