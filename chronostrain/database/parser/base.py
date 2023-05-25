import os
from abc import abstractmethod
from pathlib import Path
import pickle

from .. import StrainDatabase


class AbstractDatabaseParser(object):
    def __init__(self, db_name: str, data_dir: Path):
        self.db_name = db_name
        self.data_dir = data_dir

    @abstractmethod
    def parse(self) -> StrainDatabase:
        raise NotImplementedError()

    @staticmethod
    def database_pkl_name() -> str:
        if os.name == 'nt':
            # Windows paths
            return 'database.windows.pkl'
        else:
            # Posix paths
            return 'database.posix.pkl'

    def disk_path(self) -> Path:
        """
        Certain object attributes (such as marker metadata) uses pathlib.Path, which is specific to the OS.
        Therefore, save/load each separately.

        :return: The target path to save the database.
        """
        return StrainDatabase.database_named_dir(self.data_dir, self.db_name) / AbstractDatabaseParser.database_pkl_name()

    def save_to_disk(self, db: StrainDatabase):
        pkl_path = self.disk_path()
        if not pkl_path.parent.parent.exists():
            raise FileNotFoundError(f"Data directory {pkl_path.parent.parent} does not exist!")
        pkl_path.parent.mkdir(exist_ok=True, parents=False)
        with open(pkl_path, 'wb') as f:
            pickle.dump(db.backend, f)

    def load_from_disk(self) -> StrainDatabase:
        """Default implementation: use pickle format."""
        print(self.disk_path())
        with open(self.disk_path(), 'rb') as f:
            backend = pickle.load(f)
        return StrainDatabase(
            backend=backend,
            data_dir=self.data_dir,
            name=self.db_name,
            force_refresh=False
        )

class StrainDatabaseParseError(Exception):
    pass
