
from src.utils.logger import get_logger

logger = get_logger('test_kk_log')

logger.debug("Debugging message")
logger.info("Information message")
logger.warning("Warning message")
logger.error("Error message")
logger.critical("Critical message")