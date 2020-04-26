import os
import logging
import logging.config

_config_filename = 'log_config_default.ini'


def get_logger(key='root'):
    return logging.getLogger(key)


# ============= Create logger instance ===========
__config_loaded__ = False
if not __config_loaded__:
    try:
        logging.config.fileConfig(_config_filename)
    except FileNotFoundError as e:
        path = os.path.dirname(e.filename)
        print("[logger.py] Creating file path ", path)
        os.makedirs(path, exist_ok=True)
    logger = get_logger()
    __config_loaded__ = True
