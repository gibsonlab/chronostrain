from enum import Enum, auto


class ReadType(Enum):
    PAIRED_END_1 = auto()
    PAIRED_END_2 = auto()
    SINGLE_END = auto()

    @staticmethod
    def parse_from_str(token: str) -> 'ReadType':
        if token == "paired_1":
            return ReadType.PAIRED_END_1
        elif token == "paired_2":
            return ReadType.PAIRED_END_2
        elif token == "single":
            return ReadType.SINGLE_END
        else:
            raise ValueError(f"Unrecognized read type token `{token}`.")
