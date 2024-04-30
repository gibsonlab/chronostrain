from typing import Iterator

from .base import AbstractConfig
import jax


class EngineConfig(AbstractConfig):

    def __init__(self, cfg: dict):
        super().__init__("Engine", cfg)
        self.prng_key_initial = jax.random.PRNGKey(self.get_int("PRNG_Key"))
        self.prng_key = jax.random.PRNGKey(self.get_int("PRNG_Key"))
        self.dtype = self.get_str("DTYPE")

    def reset_prng(self):
        self.prng_key = self.prng_key_initial

    def generate_prng_keys(self, num_keys: int = 1) -> Iterator[jax.random.PRNGKey]:
        new_keys = jax.random.split(self.prng_key, num=num_keys + 1)
        self.prng_key = new_keys[0]
        for k in new_keys[1:]:
            yield k
