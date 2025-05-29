import logging
import sys
import os
from datetime import datetime

# Create logger
logger = logging.getLogger("ga8ed-api")

# Set log level from environment variable, default to INFO
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logger.setLevel(getattr(logging, log_level))

# Create handlers
console_handler = logging.StreamHandler(sys.stdout)
file_handler = logging.FileHandler(f"ga8ed_api_{datetime.now().strftime('%Y%m%d')}.log")

# Create formatters and add it to handlers
log_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(log_format)
file_handler.setFormatter(log_format)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Log the current log level
logger.info(f"Logger initialized with level: {log_level}")


def log_error(error: Exception, context: str = ""):
    """Log an error with context"""
    logger.error(f"{context}: {str(error)}", exc_info=True)


def log_info(message: str):
    """Log an info message"""
    logger.info(message)


def log_debug(message: str):
    """Log a debug message"""
    logger.debug(message)


def log_warning(message: str):
    """Log a warning message"""
    logger.warning(message)
