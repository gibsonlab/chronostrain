"""
    data_cache.py
    Author: Youn Kim
    Date: 11/30/2020

    A general-purpose utility to encapsulate functions meant for intermediate computation.
    Generates a cache key to avoid re-computation in future runs.
"""
import os
from typing import Callable
from chronostrain.util.io.logger import logger
from chronostrain import cfg
import pickle
import hashlib


class CachedComputation(object):
    def __init__(self,
                 fn: Callable,
                 cache_tag: str,
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
        self.cache_hex = hashlib.md5(cache_tag.encode('utf-8')).hexdigest()
        self.cache_dir = os.path.join(self.cache_root_dir, self.cache_hex)
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def call(self, filename: str, *args, **kwargs):
        cache_path = os.path.join(self.cache_dir, filename)
        # Try to retrieve from cache.
        try:
            data = self.loader(cache_path)
            logger.debug("[Cache {}] Loaded pre-computed file {}.".format(self.cache_hex, cache_path))
            return data
        except FileNotFoundError:
            logger.debug("[Cache {}] Could not load cached file {}. Recomputing.".format(self.cache_hex, cache_path))

        data = self.fn(*args, **kwargs)
        self.saver(cache_path, data)
        logger.debug("[Cache {}] Saved {}.".format(self.cache_hex, cache_path))
        return data
