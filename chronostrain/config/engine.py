from typing import Iterator

from .base import AbstractConfig, ConfigurationParseError
import jax


class EngineConfig(AbstractConfig):

    def __init__(self, cfg: dict):
        super().__init__("PyTorch", cfg)
        device_token = self.get_str("DEVICE")
        if device_token == "cuda":
            if len(jax.devices("cuda") == 0):
                raise RuntimeError("No CUDA devices found.")
            self.device = jax.devices("cuda")[0]
        elif device_token == "cpu":
            self.device = jax.devices("cpu")[0]
        else:
            raise ConfigurationParseError(
                "Field `DEVICE`:Invalid or unsupported device token `{}`".format(device_token)
            )

        self.prng_key = jax.random.PRNGKey(self.get_int("PRNGKey"))

    def generate_prng_keys(self, num_keys: int = 1) -> Iterator[jax.random.PRNGKey]:
        new_keys = jax.random.split(self.prng_key, num=num_keys + 1)
        self.prng_key = new_keys[0]
        for k in new_keys[1:]:
            yield k
