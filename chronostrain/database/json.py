from typing import Union
from pathlib import Path

from .database import StrainDatabase
from .backend import PandasAssistedBackend
from .parser import JSONParser, AbstractDatabaseParser

from chronostrain.logging import create_logger
logger = create_logger(__name__)


class JSONStrainDatabase(StrainDatabase):
    def __init__(self,
                 entries_file: Union[str, Path],
                 marker_max_len: int,
                 data_dir: Path,
                 force_refresh: bool = False):
        if isinstance(entries_file, str):
            entries_file = Path(entries_file)

        self.entries_file = entries_file
        parser = JSONParser(entries_file,
                            data_dir,
                            marker_max_len,
                            force_refresh)
        backend = PandasAssistedBackend()
        super().__init__(
            parser=parser,
            backend=backend,
            data_dir=data_dir,
            name=parser.entries_file.stem
        )

    def pickle_is_stale(self):
        if not self.pickle_path.exists():
            return True
        else:
            return self.entries_file.stat().st_mtime > self.pickle_path.stat().st_mtime

    def initialize(self, parser: AbstractDatabaseParser, force_refresh: bool):
        """
        Auto-invokes database save() and load() if available, checking for staleness by comparing against
        the last modification timestamp.
        """
        if self.pickle_is_stale():
            logger.info("Populating database.")
            super().initialize(parser, force_refresh)
            self.save_to_disk()
            logger.debug(f"Saved database to {self.pickle_path}.")
        else:
            logger.debug(f"Loaded database from disk ({self.pickle_path}).")
            self.load_from_disk()
