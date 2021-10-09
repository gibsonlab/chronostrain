import os
import sys
from pathlib import Path
import errno
import logging
import logging.config
import logging.handlers


__env_key__ = "CHRONOSTRAIN_LOG_INI"
__ini_path__ = os.getenv(__env_key__, "log_config.ini")

from typing import Callable


class LoggingLevelFilter(logging.Filter):
    def __init__(self, levels):
        super().__init__()
        self.levels = levels

    def filter(self, rec):
        return rec.levelno in self.levels


class MakeDirTimedRotatingFileHandler(logging.handlers.TimedRotatingFileHandler):
    """
    A class which calls makedir() on the specified file path.
    """
    def __init__(self,
                 filename,
                 when='h',
                 interval=1,
                 backupCount=0,
                 encoding=None,
                 delay=False,
                 utc=False,
                 atTime=None):
        path = Path(filename).resolve()
        MakeDirTimedRotatingFileHandler.mkdir_path(path.parent)
        super().__init__(filename=filename,
                         when=when,
                         interval=interval,
                         backupCount=backupCount,
                         encoding=encoding,
                         delay=delay,
                         utc=utc,
                         atTime=atTime)

    @staticmethod
    def mkdir_path(path):
        """http://stackoverflow.com/a/600612/190597 (tzot)"""
        try:
            os.makedirs(path, exist_ok=True)  # Python>3.2
        except TypeError:
            try:
                os.makedirs(path)
            except OSError as exc:  # Python >2.5
                if exc.errno == errno.EEXIST and Path(path).is_dir():
                    pass
                else:
                    raise


def default_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.addFilter(LoggingLevelFilter([logging.INFO, logging.DEBUG]))
    stdout_handler.setLevel(logging.DEBUG)
    stdout_formatter = logging.Formatter("[%(levelname)s - %(name)s] - %(message)s")
    stdout_handler.setFormatter(stdout_formatter)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.addFilter(LoggingLevelFilter([logging.ERROR, logging.WARNING, logging.CRITICAL]))
    stderr_handler.setLevel(logging.WARNING)
    stderr_formatter = logging.Formatter("[%(levelname)s - %(name)s] - %(message)s")
    stderr_handler.setFormatter(stderr_formatter)

    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)
    return logger


def meta_create_logger(ini_path: Path) -> Callable[[str], logging.Logger]:
    if not ini_path.exists():
        def __create_logger(module_name):
            return default_logger(module_name)

        this_logger = __create_logger(__name__)
        this_logger.debug("Using default logger (stdout, stderr).")
    else:
        def __create_logger(module_name):
            return logging.getLogger(module_name)

        this_logger = __create_logger(__name__)
        this_logger.debug("Using logging configuration {}".format(
            str(ini_path)
        ))

    return __create_logger


create_logger = meta_create_logger(Path(__ini_path__))
