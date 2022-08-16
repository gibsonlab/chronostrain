from abc import ABCMeta
from typing import Any, Dict, Iterator
from pathlib import Path


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
