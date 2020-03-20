import logging
import logging.config

_config_filename = 'log_config_default.ini'

# ============= Create logger instance ===========
__config_loaded__ = False
if not __config_loaded__:
    logging.config.fileConfig(_config_filename)
    logger = logging.getLogger()
    __config_loaded__ = True
