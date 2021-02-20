import os
from configparser import ConfigParser
from .config import ConfigurationParseError
from .config import AbstractConfig, ChronostrainConfig, DatabaseConfig, ModelConfig


def _load(ini_path) -> ChronostrainConfig:
    if not os.path.exists(ini_path):
        raise FileNotFoundError("No configuration INI file found. "
                                "Create a `chronostrain.ini` file, or set the `CHRONOSTRAIN_INI` "
                                "environment variable to point to the right configuration.")

    cfg_parser = ConfigParser()
    cfg_parser.read(ini_path)
    _config = ChronostrainConfig(dict(cfg_parser))
    return _config


__ini__ = os.getenv("CHRONOSTRAIN_INI", "chronostrain.ini")
cfg = _load(__ini__)
