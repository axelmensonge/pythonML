import logging
from core.config import LOG_PATH, LOG_FORMAT, LOG_DIR

LOG_DIR.mkdir(parents=True, exist_ok=True)

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(LOG_FORMAT)

        file_handler = logging.FileHandler(LOG_PATH, mode="a", encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger