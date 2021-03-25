import os
from configparser import ConfigParser
from chronostrain.util.logger import logger
from .config import ConfigurationParseError
from .config import AbstractConfig, ChronostrainConfig, DatabaseConfig, ModelConfig

__env_key__ = "CHRONOSTRAIN_INI"
__ini__ = os.getenv(__env_key__, "chronostrain.ini")


def _load(ini_path) -> ChronostrainConfig:
    if not os.path.exists(ini_path):
        raise FileNotFoundError("No configuration INI file found. Create a `chronostrain.ini` file, or set the `{}` environment variable to point to the right configuration.".format(__env_key__))

    cfg_parser = ConfigParser()
    cfg_parser.read(ini_path)
    _config = ChronostrainConfig(dict(cfg_parser))
    logger.debug("Loaded chronostrain INI from {}.".format(ini_path))
    return _config


# ============= Create configuration instance. ===========
cfg = _load(__ini__)
