import os
from configparser import SafeConfigParser
from chronostrain.util.logger import logger
from .config import ConfigurationParseError
from .config import AbstractConfig, ChronostrainConfig, DatabaseConfig, ModelConfig

__env_key__ = "CHRONOSTRAIN_INI"
__ini__ = os.getenv(__env_key__, "chronostrain.ini")


def _load(ini_path) -> ChronostrainConfig:
    if not os.path.exists(ini_path):
        raise FileNotFoundError("No configuration INI file found. Create a `chronostrain.ini` file, or set the `{}` environment variable to point to the right configuration.".format(__env_key__))

    cfg_parser = SafeConfigParser()
    cfg_parser.read(ini_path)

    config_dict = {}
    for section in cfg_parser.sections():
        config_dict[section] = {
            item.upper(): cfg_parser.get(section, item, vars=os.environ)
            for item in cfg_parser.options(section)
        }
    _config = ChronostrainConfig(config_dict)
    logger.debug("Loaded chronostrain INI from {}.".format(ini_path))
    return _config


# ============= Create configuration instance. ===========
cfg = _load(__ini__)
