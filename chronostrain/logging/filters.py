import logging


class LoggingLevelFilter(logging.Filter):
    def __init__(self, levels):
        super().__init__()
        self.levels = levels

    def filter(self, rec):
        return rec.levelno in self.levels
