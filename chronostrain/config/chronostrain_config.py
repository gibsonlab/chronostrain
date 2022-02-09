import os
import importlib
from abc import ABCMeta
from typing import Any, Dict, Iterator
from pathlib import Path
from configparser import ConfigParser

import torch
import chronostrain

from .logging import create_logger
logger = create_logger(__name__)


class ConfigurationParseError(Exception):
    pass


class AbstractConfig(metaclass=ABCMeta):
    def __init__(self, name: str, cfg_dict: Dict[str, str]):
        self.name = name
        self.cfg_dict = cfg_dict

    @staticmethod
    def key_variants(key: str) -> Iterator[str]:
        yield key
        yield key.upper()
        yield key.lower()

    def get_item(self, key: str) -> Any:
        for key_to_try in self.key_variants(key):
            try:
                return self.cfg_dict[key_to_try]
            except KeyError:
                logger.warning(f"Could not find key `{key_to_try}`. Trying a variant...")
                continue
        raise ConfigurationParseError("Could not find key {} in configuration '{}'.".format(
            key,
            self.name,
        ))

    def get_str(self, key: str) -> str:
        return self.get_item(key).strip()

    def get_float(self, key: str) -> float:
        try:
            return float(self.get_str(key))
        except ValueError:
            raise ConfigurationParseError(
                f"Field `{key}`: Expected `float`, got value `{self.get_str(key)}`"
            )

    def get_int(self, key: str) -> int:
        try:
            return int(self.get_str(key))
        except ValueError:
            raise ConfigurationParseError(
                f"Field `{key}`: Expected `int`, got value `{self.get_str(key)}`"
            )

    def get_bool(self, key: str) -> bool:
        item = self.get_str(key)
        if item.lower() == "true":
            return True
        elif item.lower() == "false":
            return False
        else:
            raise ConfigurationParseError(
                f"Field `{key}`: Expected `float`, got value `{item}`"
            )

    def get_path(self, key: str) -> Path:
        return Path(self.get_str(key))


class DatabaseConfig(AbstractConfig):
    def __init__(self, cfg: Dict[str, str], database_kwargs: dict):
        super().__init__("Database", cfg)
        self.db_kwargs = {
            key.lower(): (value if value != 'None' else None)
            for key, value in database_kwargs.items()
        }
        self.class_name: str = self.get_str("DB_CLASS")
        self.data_dir: Path = self.get_path("DATA_DIR")
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def get_database(self, force_refresh: bool = False, **kwargs) -> "chronostrain.database.StrainDatabase":
        """
        Creates a new instance of a StrainDatabase object.
        """
        module_name, class_name = self.class_name.rsplit(".", 1)
        class_ = getattr(importlib.import_module(module_name), class_name)
        db_kwargs = self.db_kwargs.copy()
        db_kwargs["force_refresh"] = force_refresh
        for k, v in kwargs.items():
            db_kwargs[k] = v

        db_obj = class_(**db_kwargs)
        if not isinstance(db_obj, chronostrain.database.StrainDatabase):
            raise RuntimeError("Specified database class {} is not a subclass of {}".format(
                self.class_name,
                chronostrain.database.StrainDatabase.__class__.__name__
            ))
        return db_obj


class ModelConfig(AbstractConfig):
    def __init__(self, cfg: dict):
        super().__init__("Model", cfg)
        self.use_quality_scores: bool = self.get_bool("USE_QUALITY_SCORES")
        self.num_cores: int = self.get_int("NUM_CORES")
        self.cache_dir: Path = self.get_path("CACHE_DIR")
        self.sics_dof_1: float = self.get_float("SICS_DOF_1")
        self.sics_scale_1: float = self.get_float("SICS_SCALE_1")
        self.sics_dof: float = self.get_float("SICS_DOF")
        self.sics_scale: float = self.get_float("SICS_SCALE")
        self.use_sparse: bool = self.get_bool("SPARSE_MATRICES")
        self.frag_len_negbin_n: float = self.get_float("FRAG_LEN_NB_N")
        self.frag_len_negbin_p: float = self.get_float("FRAG_LEN_NB_P")


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
        super().__init__("PyTorch", cfg)
        device_token = self.get_str("DEVICE")
        if device_token == "cuda":
            self.device = torch.device("cuda")
        elif device_token == "cpu":
            self.device = torch.device("cpu")
        else:
            raise ConfigurationParseError(
                "Field `DEVICE`:Invalid or unsupported device token `{}`".format(device_token)
            )

        dtype_str = self.get_str("DEFAULT_DTYPE")
        try:
            self.default_dtype = TorchConfig.torch_dtypes[dtype_str]
        except KeyError:
            raise ConfigurationParseError("Invalid dtype token `{}`.".format(
                dtype_str
            ))
        torch.set_default_dtype(self.default_dtype)


class ExternalToolsConfig(AbstractConfig):
    def __init__(self, cfg: dict):
        super().__init__("ExternalTools", cfg)
        self.pairwise_align_cmd = self.get_str("PAIRWISE_ALN_BACKEND")
        self.glopp_path = self.get_str("GLOPP")


class EntrezConfig(AbstractConfig):
    def __init__(self, cfg: dict):
        super().__init__("Entrez", cfg)
        self.email = self.get_str("EMAIL")


class ChronostrainConfig(AbstractConfig):
    def __init__(self, cfg: dict):
        super().__init__("ChronoStrain", cfg)
        self.database_cfg: DatabaseConfig = DatabaseConfig(
            cfg=self.get_item("Database"),
            database_kwargs=self.get_item("Database.args")
        )
        self.model_cfg: ModelConfig = ModelConfig(self.get_item("Model"))
        self.torch_cfg: TorchConfig = TorchConfig(self.get_item("PyTorch"))
        self.external_tools_cfg: ExternalToolsConfig = ExternalToolsConfig(self.get_item("ExternalTools"))
        self.entrez_cfg: EntrezConfig = EntrezConfig(self.get_item("Entrez"))


def _config_load(ini_path: str) -> ChronostrainConfig:
    if not Path(ini_path).exists():
        raise FileNotFoundError(
            "Config INI path `{}` invalid. Create a `chronostrain.ini` file, or set the `{}` environment "
            "variable to point to the right configuration.".format(
                str(ini_path),
                __env_key__
            )
        )

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
