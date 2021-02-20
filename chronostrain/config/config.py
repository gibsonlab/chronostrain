from abc import ABCMeta, abstractmethod
from typing import Tuple, Any
from pathlib import Path

import torch


class ConfigurationParseError(BaseException):
    pass


class AbstractConfig(metaclass=ABCMeta):
    def __init__(self, name: str):
        self.name = name

    def parse(self, cfg: dict):
        try:
            return self.parse_impl(cfg)
        except KeyError as e:
            raise ConfigurationParseError("KeyError in config `{}`: {}".format(
                self.name,
                str(e)
            ))

    @abstractmethod
    def parse_impl(self, cfg: dict):
        raise NotImplementedError()


class DatabaseConfig(AbstractConfig):
    def __init__(self, cfg: dict):
        super().__init__("Database")
        self.database_path, self.data_dir, self.marker_max_len = self.parse(cfg)

    def parse_impl(self, cfg: dict) -> Tuple[str, str, int]:
        path = cfg["DB_PATH"]

        datadir = cfg["DATA_DIR"]
        Path(datadir).mkdir(parents=True, exist_ok=True)

        m_str = cfg["MARKER_MAX_LEN"].strip()
        try:
            marker_max_len = int(m_str)
        except ValueError:
            raise ConfigurationParseError("Token `MARKER_MAX_LEN`: Expected integer, got `{}`".format(m_str))
        return path, datadir, marker_max_len


class ModelConfig(AbstractConfig):
    def __init__(self, cfg: dict):
        super().__init__("Model")
        self.use_quality_scores, self.num_cores = self.parse(cfg)

    def parse_impl(self, cfg: dict) -> Tuple[bool, int]:
        q_token = cfg["USE_QUALITY_SCORES"].strip().lower()
        if q_token == "true":
            use_quality_scores = True
        elif q_token == "false":
            use_quality_scores = False
        else:
            raise ConfigurationParseError("Field `USE_QUALITY_SCORES`: Expected `true` or `false`, got `{}`".format(q_token))

        n_cores_token = cfg["NUM_CORES"]
        try:
            n_cores = int(n_cores_token.strip())
        except ValueError:
            raise ConfigurationParseError("Field `NUM_CORES`: Expected int, got `{}`".format(n_cores_token))

        return use_quality_scores, n_cores


class TorchConfig(AbstractConfig):
    def __init__(self, cfg: dict):
        super().__init__("PyTorch")
        (self.device, self.default_dtype) = self.parse(cfg)

        # Initialize torch settings.
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

        try:
            dtype = torch_dtypes[self.default_dtype]
            torch.set_default_dtype(dtype)
        except KeyError:
            raise ConfigurationParseError("Invalid dtype token `{}`.".format(
                self.default_dtype
            ))

    def parse_impl(self, cfg: dict) -> Tuple[torch.device, Any]:
        device_token = cfg["DEVICE"]
        if device_token == "cuda":
            device = torch.device("cuda")
        elif device_token == "cpu":
            device = torch.device("cpu")
        else:
            raise ConfigurationParseError("Field `DEVICE`:Invalid or unsupported device token `{}`".format(device_token))
        return device, cfg["DEFAULT_DTYPE"]


class ChronostrainConfig(AbstractConfig):
    def __init__(self, cfg: dict):
        super().__init__("ChronoStrain")

        (self.database_cfg,
         self.model_cfg,
         self.torch_cfg) = self.parse(cfg)

    def parse_impl(self, cfg: dict) -> Tuple[DatabaseConfig, ModelConfig, TorchConfig]:
        return DatabaseConfig(cfg["Database"]), ModelConfig(cfg["Model"]), TorchConfig(cfg["PyTorch"])
