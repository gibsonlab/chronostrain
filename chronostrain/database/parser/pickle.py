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

    def disk_path(self) -> Path:
        if self.direct_pkl_path is None:
            return super().disk_path()
        else:
            return self.direct_pkl_path

    def parse(self) -> StrainDatabase:
        logger.debug("Loading database instance from {}.".format(self.disk_path()))
        return self.load_from_disk()
