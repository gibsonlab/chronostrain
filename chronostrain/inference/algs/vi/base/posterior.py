from abc import abstractmethod, ABC, ABCMeta
from pathlib import Path
from typing import Optional

import jax.numpy as np
from .constants import GENERIC_SAMPLE_TYPE, GENERIC_PARAM_TYPE
import importlib


class AbstractPosterior(metaclass=ABCMeta):
    @abstractmethod
    def abundance_sample(self, num_samples: int = 1) -> np.ndarray:
        """
        Returns a sample from this posterior distribution.
        :param num_samples: the number of samples (N).
        :return: A time-indexed, simplex-valued (T x N x S) abundance tensor.
        """
        pass

    @abstractmethod
    def log_likelihood(self, x: np.ndarray) -> float:
        pass

    @abstractmethod
    def save(self, target_path: Path):
        pass


class AbstractReparametrizedPosterior(AbstractPosterior, ABC):
    def __init__(self, params: Optional[GENERIC_PARAM_TYPE] = None):
        if params is None:
            self.parameters = self.initial_params()
        else:
            self.parameters = params

    @abstractmethod
    def initial_params(self) -> GENERIC_PARAM_TYPE:
        raise NotImplementedError()

    def log_likelihood(self, samples: np.ndarray, params: GENERIC_PARAM_TYPE = None) -> np.ndarray:
        pass

    @abstractmethod
    def entropy(self, params: GENERIC_PARAM_TYPE, *args) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def random_sample(self, num_samples: int) -> GENERIC_SAMPLE_TYPE:
        """
        Return randomized samples (before reparametrization.)
        :param num_samples:
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def get_parameters(self) -> GENERIC_PARAM_TYPE:
        raise NotImplementedError()

    @abstractmethod
    def set_parameters(self, params: GENERIC_PARAM_TYPE):
        """
        Store the value of these params internally as the state of this posterior.
        :param params: A list of parameter arrays (the implementation should decide the ordering.)
        :return:
        """
        pass

    def reparametrize(self, random_samples: GENERIC_SAMPLE_TYPE, params: GENERIC_PARAM_TYPE, *args) -> GENERIC_SAMPLE_TYPE:
        raise NotImplementedError()

    def save(self, path: Path):
        np.savez(
            str(path),
            **self.parameters
        )

    def load(self, path: Path):
        f = np.load(str(path))
        self.parameters = dict(f)

    @abstractmethod
    def save_class_initializer(self, path: Path):
        """
        All implementations of this method must follow this format:
        Line 1: Class path
        Line 2...n: Arg name and value, of the format "ArgName=ArgValue". Example: num_strains=100

        :param path:
        """
        raise NotImplementedError()

    @staticmethod
    def load_class_from_initializer(path: Path) -> 'AbstractReparametrizedPosterior':
        """
        Load the specified classname and kwargs from the specified path, and instantiate & return it.
        Note that this function does not load the actual parameter values.

        :return: A posterior instance extending AbstractReparametrizedPosterior.
        """
        with open(path, "rt") as f:
            full_class_path = f.readline().strip()
            parser_kwargs = {}

            for line in f:
                line = line.strip()
                tokens = line.split("=")
                if len(tokens) != 2:
                    raise ValueError("While parsing posterior metadata file {}, found an invalid kwargs line ({})".format(
                        path, line
                    ))
                key_full, value = tokens
                key_tokens = key_full.split(":")
                if len(key_tokens) != 2:
                    raise ValueError("While parsing posterior metadata file {}, found an invalid parameter key ({})".format(
                        path, key_full
                    ))

                key_name, key_type = key_tokens
                if key_type == 'int':
                    value = int(value)
                elif key_type == 'float':
                    value = float(value)
                elif key_type == 'str':
                    pass
                else:
                    raise ValueError(
                        "While parsing posterior metadata file {}, found an invalid Key type ({})".format(
                            path, key_type
                        ))
                parser_kwargs[key_name] = value

        module_name, class_name = full_class_path.rsplit(".", 1)
        class_ = getattr(importlib.import_module(module_name), class_name)
        posterior_obj = class_(**parser_kwargs)

        # Validate object.
        if not isinstance(posterior_obj, AbstractReparametrizedPosterior):
            raise RuntimeError(f"Specified posterior class {full_class_path} is not a subclass of AbstractReparametrizedPosterior")

        return posterior_obj
