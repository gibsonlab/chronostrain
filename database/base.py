from abc import abstractmethod, ABCMeta


class AbstractDatabase(metaclass=ABCMeta):
    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def get_markers(self, strain_id):
        pass


class SimpleCSVDatabase(AbstractDatabase):
    def load(self):
        # TODO load from specified CSV file
        pass

    def get_markers(self, strain_id):
        # TODO just query the strain_id as a key from dict
        pass