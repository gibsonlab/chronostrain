"""
    data_cache.py
    Author: Youn Kim
    Date: 11/30/2020

    A general-purpose utility to encapsulate functions meant for intermediate computation.
    Generates a cache key to avoid re-computation in future runs.
"""
from pathlib import Path
from typing import Callable, Optional, List, Dict
import pickle
import hashlib

from . import logger
from chronostrain.config import cfg
from chronostrain.util.filesystem import md5_checksum


class CacheTag(object):
    def __init__(self, **kwargs):
        """
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
        return hashlib.md5(str(processed_dict).encode('utf-8')).hexdigest()

    def process_item(self, item) -> str:
        if isinstance(item, list):
            return "[{}]".format(",".join(
                self.process_item(entry) for entry in item
            ))
        elif isinstance(item, Path):
            return md5_checksum(item)
        else:
            return str(item)

    def write_attributes_to_disk(self, path: Path):
        with open(path, "w") as f:
            for key, value in self.attr_dict.items():
                if isinstance(value, list):
                    print("{}:".format(key), file=f)
                    for item in value:
                        print("\t{}".format(item), file=f)
                else:
                    print("{}: {}".format(key, value), file=f)


class CachedComputation(object):
    def __init__(self,
                 fn: Callable,
                 filename: str,
                 cache_tag: CacheTag,
                 args: Optional[List] = [],
                 kwargs: Optional[Dict] = {},
                 save: Optional[Callable] = None,
                 load: Optional[Callable] = None,):
        """
        :param save: A function or Callable which takes (1) a filepath and (2) a python object as input to
        save the designated object to the specified file.
        :param load: A function or Callable which takes a filepath as input to load some object from the file.
        """
        self.fn = fn
        self.cache_tag = cache_tag

        self.cache_dir = cfg.model_cfg.cache_dir / self.cache_tag.encoding
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.cache_dir / filename

        self.args = args
        self.kwargs = kwargs

        self.saver = save
        self.loader = load
        if self.saver is None:
            def save_(path, obj):
                with open(path, "wb") as f:
                    pickle.dump(obj, f)
            self.saver = save_
        if self.loader is None:
            def load_(path):
                with open(path, "rb") as f:
                    return pickle.load(f)
            self.loader = load_

    def call(self):
        # Try to retrieve from cache.
        try:
            data = self.loader(self.cache_path)
            logger.debug("[Cache {}] Loaded pre-computed file {}.".format(
                self.cache_tag.encoding, self.cache_path
            ))
            return data
        except FileNotFoundError:
            logger.debug("[Cache {}] Could not load cached file {}. Recomputing.".format(
                self.cache_tag.encoding, self.cache_path
            ))

        self.cache_tag.write_attributes_to_disk(self.cache_dir / "attributes.txt")
        data = self.fn(*self.args, **self.kwargs)
        self.saver(self.cache_path, data)

        logger.debug("[Cache {}] Saved {}.".format(
            self.cache_tag.encoding, self.cache_path
        ))
        return data
