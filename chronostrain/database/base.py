from abc import abstractmethod, ABCMeta
from typing import List
from chronostrain.model.bacteria import Strain


class AbstractStrainDatabase(metaclass=ABCMeta):
    def __init__(self):
        self.__load__()

    @abstractmethod
    def __load__(self):
        pass

    @abstractmethod
    def get_strain(self, strain_id: str) -> Strain:
        pass

    @abstractmethod
    def all_strains(self) -> List[Strain]:
        pass

    def get_strains(self, strain_ids: List[str]) -> List[Strain]:
        return [self.get_strain(s_id) for s_id in strain_ids]


class StrainEntryError(BaseException):
    pass


class StrainNotFoundError(BaseException):
    def __init__(self, strain_id):
        self.strain_id = strain_id
        super().__init__("Strain id `{}` not found in database.".format(strain_id))
