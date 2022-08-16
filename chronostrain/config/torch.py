from .base import AbstractConfig, ConfigurationParseError
import torch


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
