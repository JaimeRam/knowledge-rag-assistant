import logging
import sys
from pythonjsonlogger import jsonlogger


def setup_logging(level: str = "INFO"):
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # JSON formatter for structured logging
    formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(name)s %(levelname)s %(message)s'
    )
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger
