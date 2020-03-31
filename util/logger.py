import logging
import logging.config

_config_filename = 'log_config_default.ini'


def get_logger(key='root'):
    return logging.getLogger(key)


# ============= Create logger instance ===========
__config_loaded__ = False
if not __config_loaded__:
    logging.config.fileConfig(_config_filename)
    logger = get_logger()
    __config_loaded__ = True
