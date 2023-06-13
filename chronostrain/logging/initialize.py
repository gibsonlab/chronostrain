import os
import sys
from pathlib import Path
from typing import Callable

import logging
import logging.config
import logging.handlers
from .filters import LoggingLevelFilter
from .handlers import MakeDirTimedRotatingFileHandler


__env_key__ = "CHRONOSTRAIN_LOG_INI"
__ini_path__ = os.getenv(__env_key__, "log_config.ini")


def default_logger(name: str):
    logger = logging.getLogger(name=name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.addFilter(LoggingLevelFilter([logging.INFO, logging.DEBUG]))
    stdout_handler.setLevel(logging.DEBUG)
    stdout_formatter = logging.Formatter("%(asctime)s [%(levelname)s - %(name)s] - %(message)s")
    stdout_handler.setFormatter(stdout_formatter)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.addFilter(LoggingLevelFilter([logging.ERROR, logging.WARNING, logging.CRITICAL]))
    stderr_handler.setLevel(logging.WARNING)
    stderr_formatter = logging.Formatter("%(asctime)s [%(levelname)s - %(name)s] - %(message)s")
    stderr_handler.setFormatter(stderr_formatter)

    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)
    return logger


def meta_create_logger(ini_path: Path) -> Callable[[str], logging.Logger]:
    logging.handlers.MakeDirTimedRotatingFileHandler = MakeDirTimedRotatingFileHandler
    if not ini_path.exists():
        def __create_logger(module_name):
            return default_logger(name=module_name)

        this_logger = __create_logger(__name__)
        this_logger.debug("Using default logger (stdout, stderr).")
    else:
        def __create_logger(module_name):
            return logging.getLogger(name=module_name)

        logging.config.fileConfig(ini_path)
        this_logger = __create_logger(__name__)
        this_logger.debug("Using logging configuration {}".format(
            str(ini_path)
        ))

    return __create_logger


create_logger = meta_create_logger(Path(__ini_path__))
logging.getLogger("jax").setLevel(logging.INFO)
