from enum import Enum


class ResultStatus(Enum):
    INITIALIZED = -1
    NO_RESULT = 0
    GOOD = 1
    WARN = 2
    BAD = 3

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented
