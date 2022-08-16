import os
import logging.handlers
import errno
from pathlib import Path


class MakeDirTimedRotatingFileHandler(logging.handlers.TimedRotatingFileHandler):
    """
    A class which calls makedir() on the specified file path.
    """
    def __init__(self,
                 filename,
                 when='D',
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
