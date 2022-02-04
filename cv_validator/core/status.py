from enum import Enum, auto


class CheckStatus(Enum):
    INITIALIZED = auto()
    DONE = auto()


class ResultStatus(Enum):
    NO_RESULT = auto()
    GOOD = auto()
    WARN = auto()
    BAD = auto()
