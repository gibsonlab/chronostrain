from typing import Dict
from pathlib import Path
import importlib

import chronostrain.database
from .base import AbstractConfig


class DatabaseConfig(AbstractConfig):
    def __init__(self, cfg: Dict[str, str], database_kwargs: dict):
        super().__init__("Database", cfg)
        self.db_kwargs = {
            key.lower(): (value if value != 'None' else None)
            for key, value in database_kwargs.items()
        }
        self.class_name: str = self.get_str("DB_CLASS")
        self.data_dir: Path = self.get_path("DB_DATA_DIR")
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def get_database(self, **kwargs) -> "chronostrain.database.StrainDatabase":
        """
        Creates a new instance of a StrainDatabase object.
        """
        module_name, class_name = self.class_name.rsplit(".", 1)
        class_ = getattr(importlib.import_module(module_name), class_name)
        db_kwargs = self.db_kwargs.copy()
        for k, v in kwargs.items():
            db_kwargs[k] = v

        db_kwargs['data_dir'] = self.data_dir
        db_obj = class_(**db_kwargs)
        if not isinstance(db_obj, chronostrain.database.StrainDatabase):
            raise RuntimeError("Specified database class {} is not a subclass of {}".format(
                self.class_name,
                chronostrain.database.StrainDatabase.__class__.__name__
            ))
        return db_obj
