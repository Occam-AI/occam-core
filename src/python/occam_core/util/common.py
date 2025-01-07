import logging
import os


def get_logger(verbose: bool = False):
    logger = logging.getLogger(__name__)
    if verbose:
        logger.setLevel(logging.DEBUG)
    elif (log_level := os.getenv('OCCAM_LOG_LEVEL')) is not None:
        level = logging.getLevelName(log_level)
        logger.setLevel(level)
    else:
        logger.setLevel(logging.INFO)
    if not logger.handlers:
        # If there are no handlers, then we can add one
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger
