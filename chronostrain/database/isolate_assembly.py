from typing import Union
from pathlib import Path

from .database import StrainDatabase
from .backend import PandasAssistedBackend
from .parser import IsolateAssemblyParser, AbstractDatabaseParser

from chronostrain.logging import create_logger
logger = create_logger(__name__)


class IsolateAssemblyDatabase(StrainDatabase):
    def __init__(
            self,
            db_name: str,
            specs: Union[str, Path],
            data_dir: Path
    ):
        if isinstance(specs, str):
            specs = Path(specs)
        self.specification = specs
        parser = IsolateAssemblyParser(self.specification)
        backend = PandasAssistedBackend()

        super().__init__(
            parser=parser,
            backend=backend,
            data_dir=data_dir,
            name=db_name
        )

    def pickle_is_stale(self):
        if not self.pickle_path.exists():
            return True
        else:
            return self.specification.stat().st_mtime > self.pickle_path.stat().st_mtime

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
