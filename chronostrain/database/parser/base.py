from abc import abstractmethod
from pathlib import Path
from typing import Iterator

from chronostrain import cfg
from chronostrain.model import Strain


class AbstractDatabaseParser(object):
    @abstractmethod
    def strains(self) -> Iterator[Strain]:
        pass


class StrainDatabaseParseError(BaseException):
    pass
