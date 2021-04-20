import os
import importlib
from abc import ABCMeta, abstractmethod
from typing import Tuple, Any
from pathlib import Path
from configparser import SafeConfigParser

import torch
import chronostrain

from . import logger


class ConfigurationParseError(BaseException):
    pass


class AbstractConfig(metaclass=ABCMeta):
    def __init__(self, name: str):
        self.name = name

    def parse(self, cfg: dict):
        try:
            return self.parse_impl(cfg)
        except KeyError as e:
            raise ConfigurationParseError("Could not find key {} in configuration '{}'.".format(
                str(e),
                self.name,
            ))

    @abstractmethod
    def parse_impl(self, cfg: dict):
        raise NotImplementedError()


class DatabaseConfig(AbstractConfig):
    def __init__(self, cfg: dict, args_cfg: dict):
        super().__init__("Database")
        self.args_cfg = args_cfg
        tokens = self.parse(cfg)
        self.class_name: str = tokens[0]
        self.kwargs: dict = tokens[1]
        self.data_dir: Path = tokens[2]

    def parse_impl(self, cfg: dict) -> Tuple[str, dict, Path]:
        class_name = cfg["DB_CLASS"]

        datadir = Path(cfg["DATA_DIR"])
        datadir.mkdir(parents=True, exist_ok=True)

        kwargs = {
            key.lower(): (value if value != 'None' else None)
            for key, value in self.args_cfg.items()
        }

        return class_name, kwargs, datadir

    def get_database(self, force_refresh: bool = False):
        module_name, class_name = self.class_name.rsplit(".", 1)
        class_ = getattr(importlib.import_module(module_name), class_name)
        db_kwargs = self.kwargs.copy()
        db_kwargs["force_refresh"] = force_refresh
        db_obj = class_(**db_kwargs)
        if not isinstance(db_obj, chronostrain.database.AbstractStrainDatabase):
            raise RuntimeError("Specified database class {} is not a subclass of {}".format(
                self.class_name,
                chronostrain.database.AbstractStrainDatabase.__class__.__name__
            ))
        return db_obj


class ModelConfig(AbstractConfig):
    def __init__(self, cfg: dict):
        super().__init__("Model")
        tokens = self.parse(cfg)
        self.use_quality_scores: bool = tokens[0]
        self.num_cores: int = tokens[1]
        self.cache_dir: Path = tokens[2]
        self.sics_dof_1: float = tokens[3]
        self.sics_scale_1: float = tokens[4]
        self.sics_dof: float = tokens[5]
        self.sics_scale: float = tokens[6]

    def parse_impl(self, cfg: dict) -> Tuple[bool, int, Path, float, float, float, float]:
        q_token = cfg["USE_QUALITY_SCORES"].strip().lower()
        if q_token == "true":
            use_quality_scores = True
        elif q_token == "false":
            use_quality_scores = False
        else:
            raise ConfigurationParseError(
                "Field `USE_QUALITY_SCORES`: Expected `true` or `false`, got `{}`".format(q_token)
            )

        try:
            n_cores = int(cfg["NUM_CORES"].strip())
        except ValueError:
            raise ConfigurationParseError(
                "Field `NUM_CORES`: Expected int, got `{}`".format(cfg["NUM_CORES"])
            )

        cache_dir = Path(cfg["CACHE_DIR"])

        try:
            sics_dof_1 = float(cfg["SICS_DOF_1"].strip())
        except ValueError:
            raise ConfigurationParseError(
                "Field `SICS_DOF_1`: Expect float, got `{}`".format(cfg["SICS_DOF_1"])
            )

        try:
            sics_scale_1 = float(cfg["SICS_SCALE_1"].strip())
        except ValueError:
            raise ConfigurationParseError(
                "Field `SICS_SCALE_1`: Expect float, got `{}`".format(cfg["SICS_SCALE_1"])
            )

        try:
            sics_dof = float(cfg["SICS_DOF"].strip())
        except ValueError:
            raise ConfigurationParseError(
                "Field `SICS_DOF`: Expect float, got `{}`".format(cfg["SICS_DOF"])
            )

        try:
            sics_scale = float(cfg["SICS_SCALE"].strip())
        except ValueError:
            raise ConfigurationParseError(
                "Field `SICS_SCALE`: Expect float, got `{}`".format(cfg["SICS_SCALE"])
            )

        return use_quality_scores, n_cores, cache_dir, sics_dof_1, sics_scale_1, sics_dof, sics_scale


class TorchConfig(AbstractConfig):
    torch_dtypes = {
        "float": torch.float,
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "double": torch.double,
        "bfloat16": torch.bfloat16,
        "half": torch.half,
        "uint8": torch.uint8,
        "int": torch.int,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "short": torch.short,
        "long": torch.long,
        "complex32": torch.complex32,
        "complex64": torch.complex64,
        "complex128": torch.complex128,
        "cfloat": torch.cfloat,
        "cdouble": torch.cdouble,
        "quint8": torch.quint8,
        "qint8": torch.qint8,
        "qint32": torch.qint32,
        "bool": torch.bool
    }

    def __init__(self, cfg: dict):
        super().__init__("PyTorch")
        (self.device, self.default_dtype) = self.parse(cfg)
        torch.set_default_dtype(self.default_dtype)

    def parse_impl(self, cfg: dict) -> Tuple[torch.device, Any]:
        device_token = cfg["DEVICE"]
        if device_token == "cuda":
            device = torch.device("cuda")
        elif device_token == "cpu":
            device = torch.device("cpu")
        else:
            raise ConfigurationParseError(
                "Field `DEVICE`:Invalid or unsupported device token `{}`".format(device_token)
            )

        dtype_str = cfg["DEFAULT_DTYPE"]

        try:
            default_dtype = TorchConfig.torch_dtypes[dtype_str]
        except KeyError:
            raise ConfigurationParseError("Invalid dtype token `{}`.".format(
                dtype_str
            ))

        return device, default_dtype


class FilteringConfig(AbstractConfig):
    def __init__(self, cfg: dict):
        super().__init__("Filtering")
        self.align_cmd = self.parse(cfg)

    def parse_impl(self, cfg: dict) -> str:
        return cfg["ALIGNER_CMD"]


class ChronostrainConfig(AbstractConfig):
    def __init__(self, cfg: dict):
        super().__init__("ChronoStrain")
        database_cfg, model_cfg, torch_cfg, filter_cfg = self.parse(cfg)
        self.database_cfg: DatabaseConfig = database_cfg
        self.model_cfg: ModelConfig = model_cfg
        self.torch_cfg: TorchConfig = torch_cfg
        self.filter_cfg: FilteringConfig = filter_cfg

    def parse_impl(self, cfg: dict) -> Tuple[DatabaseConfig, ModelConfig, TorchConfig, FilteringConfig]:
        return (
            DatabaseConfig(cfg["Database"], cfg["Database.args"]),
            ModelConfig(cfg["Model"]),
            TorchConfig(cfg["PyTorch"]),
            FilteringConfig(cfg["Filtering"])
        )


def _config_load(ini_path) -> ChronostrainConfig:
    if not Path(ini_path).exists():
        raise FileNotFoundError(
            "No configuration INI file found. Create a `chronostrain.ini` file, or set the `{}` environment "
            "variable to point to the right configuration.".format(
                __env_key__
            )
        )

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


__env_key__ = "CHRONOSTRAIN_INI"
__ini__ = os.getenv(
    key=__env_key__,
    default=str(Path(chronostrain.__file__) / "chronostrain.ini")
)
cfg = _config_load(__ini__)
