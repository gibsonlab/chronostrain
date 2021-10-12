from abc import abstractmethod
from typing import Iterator

from chronostrain.model import Strain


class AbstractDatabaseParser(object):
    @abstractmethod
    def strains(self) -> Iterator[Strain]:
        pass


class StrainDatabaseParseError(BaseException):
    pass
