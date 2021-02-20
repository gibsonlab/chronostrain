import os, sys
import logging
import logging.config


__name__ = "ChronostrainLogger"
__cfg__ = os.getenv("CHRONOSTRAIN_LOG_INI", "log_config.ini")


class InfoFilter(logging.Filter):
    def filter(self, rec):
        return rec.levelno == logging.INFO


# ============= Create logger instance ===========
__config_loaded__ = False
if not __config_loaded__:
    try:
        logging.config.fileConfig(__cfg__)
        logger = logging.getLogger(__name__)
    except FileNotFoundError as e:
        print("[chronostrain.util.io.logger] Configuration `log_config.ini` not found. "
              "Create this configuration file, "
              "or specify an existing configuration using environment variable `CHRONOSTRAIN_LOG_INI`.")

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
