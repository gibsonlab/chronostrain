"""
    data_cache.py
    Author: Youn Kim
    Date: 11/30/2020

    A general-purpose utility to encapsulate functions meant for intermediate computation.
    Generates a cache key to avoid re-computation in future runs.
"""
from pathlib import Path
from typing import Callable, Optional, List, Dict, Any
import pickle
import hashlib

from chronostrain.config import cfg
from chronostrain.model.bacteria import Strain, Marker
from chronostrain.util.filesystem import md5_checksum

from chronostrain.config.logging import create_logger
logger = create_logger(__name__)


class CacheTag(object):
    def __init__(self, **kwargs):
        """
        A cache identification object, whose initialization kwargs specify the dependencies which generate the
        cache key.
        In particular, if a file path (Path object) is passed in, then the cache key is a function of the md5 checksum
        of the file contents.

        :param kwargs: Other optional kwargs to use for generating the cache key.
        If a list is passed, each item is processed recursively.
        If a Path-like instance is passed, it is processed using its MD5 checksum.
        For all other cases, the item is converted into a string.
        """
        self.attr_dict = kwargs
        self.encoding = self.generate_encoding()

    def generate_encoding(self) -> str:
        processed_dict = dict()
        for key, value in self.attr_dict.items():
            processed_dict[key] = self.process_item(value)
        return hashlib.md5(repr(processed_dict).encode('utf-8')).hexdigest()

    def process_item(self, item) -> str:
        if isinstance(item, dict):
            logger.warning("CacheTag might not properly handle dictionary attributes.")
            return str(item)
        elif isinstance(item, list):
            return "[{}]".format(",".join(
                self.process_item(entry) for entry in item
            ))
        elif isinstance(item, Path):
            return md5_checksum(item)
        else:
            return repr(item)

    def write_readable_attributes_to_disk(self, path: Path):
        def _recursive_render(o) -> str:
            if isinstance(o, dict):
                logger.warning("CacheTag might not properly handle dictionary attributes.")
                o_str = str(o)
            elif isinstance(o, list):
                o_str = "[{}]".format(",".join(
                    _recursive_render(entry) for entry in o
                ))
            elif isinstance(o, Strain):
                o_str = "Strain({}){}".format(
                    o.id,
                    _recursive_render(o.markers)
                )
            elif isinstance(o, Marker):
                o_str = "Marker:{}".format(
                    o.id
                )
            elif isinstance(o, Path):
                o_str = "<file:{}>".format(str(o))
            else:
                o_str = str(o)
            return o_str

        with open(path, "w") as f:
            for key, value in self.attr_dict.items():
                print(
                    "{}: {}".format(key, _recursive_render(value)),
                    file=f
                )

def pickle_saver(path, obj):
    """
    Saves the object to the path using pickle.dump.
    """
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def pickle_loader(path):
    """
    Loads an object using pickle.load.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


class ComputationCache(object):
    def __init__(self, cache_tag: CacheTag):
        """
        """
        self.cache_tag = cache_tag
        self.cache_dir = cfg.model_cfg.cache_dir / self.cache_tag.encoding
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("Using cache dir {}.".format(self.cache_dir))

    def call(self,
             filename: str,
             fn: Callable,
             args: Optional[List] = None,
             kwargs: Optional[Dict] = None,
             save: Callable = pickle_saver,
             load: Callable = pickle_loader,
             success_callback: Optional[Callable] = None) -> Any:
        """
        :param filename: A unique filename to use to store the cached result, relative to the cache path.
        :param fn: A function which returns the desired output of computation, with args and kwargs passed in.
        :param args: arguments to pass to the callable `fn`.
        :param kwargs: named arguments to pass to the callable `fn`.
        :param save: A function or Callable which takes (1) a filepath and (2) a python object as input to
        save the designated object to the specified file.
        :param load: A function or Callable which takes a filepath as input to load some object from the file.
        :param success_callback: A callback to invoke at the end of either save() or load(), taking the output of
        fn() as input.
        :return: The desired output of fn(*args, **kwargs).
        """
        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}

        # Try to retrieve from cache.
        cache_path = self.cache_dir / filename
        try:
            data = load(cache_path)
            logger.debug("[Cache {}] Loaded pre-computed file {}.".format(
                self.cache_tag.encoding, cache_path
            ))
            return data
        except FileNotFoundError:
            logger.debug("[Cache {}] Could not load cached file {}. Recomputing.".format(
                self.cache_tag.encoding, cache_path
            ))

        self.cache_tag.write_readable_attributes_to_disk(self.cache_dir / "attributes.txt")
        data = fn(*args, **kwargs)
        save(cache_path, data)

        logger.debug("[Cache {}] Saved {}.".format(
            self.cache_tag.encoding, cache_path
        ))

        if success_callback is not None:
            success_callback(data)

        return data
