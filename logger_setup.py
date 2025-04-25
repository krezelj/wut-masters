import os
import logging
from typing import Optional


def setup(
        logger_name: Optional[str] = None,
        filename: Optional[str] = "tmp.log", 
        stream: bool = False,
        level: int = logging.INFO):
    
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()
    logger.setLevel(level)

    if filename is not None:
        filename = f"./.logs/{filename}"
        validate_filename(filename)
        file_formatter = logging.Formatter(
            fmt='[%(asctime)-8s][%(levelname)-8s] %(name)s:%(message)s',
            datefmt='%H:%M:%S',
        )
        file_handler = logging.FileHandler(filename, mode='w')
        file_handler.setLevel(level)
        file_handler.setFormatter(file_formatter)

        logger.addHandler(file_handler)

    if stream:
        stream_formatter = logging.Formatter('%(message)s')
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(stream_formatter)

        logger.addHandler(stream_handler)

    return logger_name


def validate_filename(filename: str):
    directories = os.path.dirname(filename)
    if directories and not os.path.exists(directories):
        os.makedirs(directories)