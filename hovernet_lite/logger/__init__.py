import logging

from hovernet_lite.constants import DEFAULT_LEVEL


def get_logger(name, level=DEFAULT_LEVEL) -> logging.Logger:
    logger = logging.getLogger(name)
    c_handler = logging.StreamHandler()
    # link handler to logger
    logger.addHandler(c_handler)
    logger.setLevel(level)
    return logger
