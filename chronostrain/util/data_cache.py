"""
    data_cache.py
    Author: Youn Kim
    Date: 11/30/2020

    A general-purpose utility to encapsulate functions meant for intermediate computation.
    Generates a cache key to avoid re-computation in future runs.
"""
import os
from typing import Callable, List, Optional
import pickle
import hashlib

from . import logger
from chronostrain.config import cfg
from chronostrain.util.filesystem import md5_checksum


class CacheTag(object):
    def __init__(self,
                 file_paths: Optional[List[str]] = None,
                 **kwargs):
        """
        :param file_paths: A list of file paths to be included in the cache key, using an MD5 hash.
        :param kwargs: Other optional kwargs to use for generating the cache key.
        """
        self.attr_dict = kwargs
        self.file_paths = file_paths
        self.objects = kwargs
        for path in file_paths:
            self.objects[path] = md5_checksum(path)
        self.encoding = hashlib.md5(str(kwargs).encode('utf-8')).hexdigest()

    def write_attributes_to_disk(self, path: str):
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
                 cache_tag: CacheTag,
                 save: Callable = None,
                 load: Callable = None):
        """
        :param save: A function or Callable which takes (1) a filepath and (2) a python object as input to
        save the designated object to the specified file.
        :param load: A function or Callable which takes a filepath as input to load some object from the file.
        """
        self.fn = fn
        self.cache_root_dir = cfg.model_cfg.cache_dir
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

        self.cache_tag = cache_tag
        self.cache_dir = os.path.join(self.cache_root_dir, self.cache_tag.encoding)
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def call(self, filename: str, *args, **kwargs):
        cache_path = os.path.join(self.cache_dir, filename)

        # Try to retrieve from cache.
        try:
            data = self.loader(cache_path)
            logger.debug("[Cache {}] Loaded pre-computed file {}.".format(self.cache_tag.encoding, cache_path))
        except FileNotFoundError:
            logger.debug("[Cache {}] Could not load cached file {}. Recomputing.".format(self.cache_tag.encoding, cache_path))
            data = self.fn(*args, **kwargs)
            self.saver(cache_path, data)
            self.cache_tag.write_attributes_to_disk(os.path.join(self.cache_dir, "attributes.txt"))
            logger.debug("[Cache {}] Saved {}.".format(self.cache_tag.encoding, cache_path))

        return data
