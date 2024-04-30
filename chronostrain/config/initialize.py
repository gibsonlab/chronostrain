import os
from pathlib import Path

from configparser import ConfigParser
from .chronostrain import ChronostrainConfig
from chronostrain.logging import create_logger
logger = create_logger(__name__)


def _config_load(ini_path: str) -> ChronostrainConfig:
    if not Path(ini_path).exists():
        print(
            "Config INI path `{}` invalid. "
            "Create a configuration INI file and pass it through the CLI option (-c), "
            "or set the `{}` environment variable to point to the right configuration.".format(
                str(ini_path),
                __env_key__
            )
        )
        exit(1)

    cfg_parser = ConfigParser()
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


__env_key__ = "CHRONOSTRAIN_INI"
__ini__ = os.getenv(
    key=__env_key__,
    default=str(Path.cwd() / "chronostrain.ini")
)
cfg_instance = _config_load(__ini__)
