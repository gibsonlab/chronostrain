import os, sys
import logging
import logging.config


__env_key__ = "CHRONOSTRAIN_LOG_INI"
__name__ = "ChronostrainLogger"
__ini__ = os.getenv(__env_key__, "log_config.ini")


class InfoFilter(logging.Filter):
    def filter(self, rec):
        return rec.levelno == logging.INFO


# ============= Create logger instance ===========
__config_loaded__ = False
if not __config_loaded__:
    try:
        logging.config.fileConfig(__ini__)
        logger = logging.getLogger(__name__)
    except FileNotFoundError as e:
        raise FileNotFoundError("No logging INI file found. Create a `log_config.ini` file, or set the `{}` environment variable to point to the right configuration.".format(__env_key__))

        # Default behavior: direct all INFO/WARNING/ERROR to stdout.
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.addFilter(InfoFilter())
        stdout_handler.setLevel(logging.INFO)

        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(logging.ERROR)

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        logger.addHandler(stdout_handler)
        logger.addHandler(stderr_handler)

    __config_loaded__ = True
