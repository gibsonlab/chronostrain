from typing import Dict
from pathlib import Path
import importlib

import chronostrain.database
from .base import AbstractConfig


class DatabaseConfig(AbstractConfig):
    def __init__(self, cfg: Dict[str, str], database_kwargs: dict):
        super().__init__("Database", cfg)
        self.parser_kwargs = {
            key.lower(): (value if value != 'None' else None)
            for key, value in database_kwargs.items()
        }
        self.parser_class_name: str = self.get_str("DB_PARSER_CLASS")
        self.data_dir: Path = self.get_path("DB_DATA_DIR")
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def get_database(self, **kwargs) -> "chronostrain.database.StrainDatabase":
        """
        Creates a new instance of a StrainDatabase object.
        """
        # Instantiate parser.
        module_name, class_name = self.parser_class_name.rsplit(".", 1)
        class_ = getattr(importlib.import_module(module_name), class_name)
        parser_kwargs = self.parser_kwargs.copy()
        for k, v in kwargs.items():
            parser_kwargs[k] = v
        parser_kwargs['data_dir'] = self.data_dir
        parser_obj = class_(**parser_kwargs)

        # Validate object.
        if not isinstance(parser_obj, chronostrain.database.AbstractDatabaseParser):
            raise RuntimeError("Specified database class {} is not a subclass of {}".format(
                self.parser_class_name,
                chronostrain.database.StrainDatabase.__class__.__name__
            ))

        # Create database
        return parser_obj.parse()
