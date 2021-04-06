from abc import abstractmethod, ABCMeta
from typing import List
from chronostrain.model.bacteria import Strain


class AbstractStrainDatabase(metaclass=ABCMeta):
    def __init__(self, force_refresh: bool = False):
        self.__load__(force_refresh=force_refresh)

    @abstractmethod
    def __load__(self, force_refresh: bool = False):
        """
        Loads the database. Automatically called by the constructor __init__.
        :param force_refresh: If true, database should refresh entire index; if necessary, should
        re-download relevant files.
        :return:
        """
        pass

    @abstractmethod
    def get_strain(self, strain_id: str) -> Strain:
        pass

    @abstractmethod
    def all_strains(self) -> List[Strain]:
        pass

    @abstractmethod
    def num_strains(self) -> int:
        pass

    @abstractmethod
    def get_multifasta_file(self) -> str:
        """
        :return: A path to a multi-fasta file, containing all of the markers in the database.
        """
        pass

    def get_strains(self, strain_ids: List[str]) -> List[Strain]:
        return [self.get_strain(s_id) for s_id in strain_ids]


class StrainEntryError(BaseException):
    pass


class StrainNotFoundError(BaseException):
    def __init__(self, strain_id):
        self.strain_id = strain_id
        super().__init__("Strain id `{}` not found in database.".format(strain_id))
