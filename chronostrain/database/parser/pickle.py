from typing import Optional, Union
from pathlib import Path

from .base import AbstractDatabaseParser
from .. import StrainDatabase
from chronostrain.logging import create_logger
logger = create_logger(__name__)


class PickleParser(AbstractDatabaseParser):
    def __init__(self, db_name: str, data_dir: Path, direct_pkl_path: Optional[Union[str, Path]] = None):
        super().__init__(db_name, data_dir)
        self.direct_pkl_path = direct_pkl_path

    def parse(self) -> StrainDatabase:
        if self.direct_pkl_path is None:
            logger.debug("Loading database instance from {}.".format(self.pickle_path()))
            return self.load_from_disk()
        else:
            logger.debug("Loading database instance from {}.".format(self.direct_pkl_path))
            return self.load_from_pkl(self.direct_pkl_path)
