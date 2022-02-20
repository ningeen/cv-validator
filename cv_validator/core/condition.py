from abc import ABC, abstractmethod

from .status import ResultStatus


class BaseCondition(ABC):
    @abstractmethod
    def __call__(self, control_value: float) -> ResultStatus:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass


class NoCondition(BaseCondition):
    def __call__(self, control_value: float) -> ResultStatus:
        return ResultStatus.NO_RESULT

    @property
    def description(self) -> str:
        return "No condition needed"


class MoreThanCondition(BaseCondition):
    def __init__(self, warn_threshold: float, error_threshold: float):
        self.warn_threshold = warn_threshold
        self.error_threshold = error_threshold
        self._description: str = ""

    def __call__(self, control_value: float) -> ResultStatus:
        if control_value is None:
            return ResultStatus.NO_RESULT

        self._description = f"Control value = {control_value:.2f} "
        if control_value > self.error_threshold:
            self._description += (
                f"which is more than {self.error_threshold:.2f}"
            )
            return ResultStatus.BAD
        if control_value > self.warn_threshold:
            self._description += (
                f"which is more than {self.warn_threshold:.2f}"
            )
            return ResultStatus.WARN
        self._description += f"which is less than {self.warn_threshold:.2f}"
        return ResultStatus.GOOD

    @property
    def description(self) -> str:
        return self._description


class LessThanCondition(BaseCondition):
    def __init__(self, warn_threshold: float, error_threshold: float):
        self.warn_threshold = warn_threshold
        self.error_threshold = error_threshold
        self._description: str = ""

    def __call__(self, control_value: float) -> ResultStatus:
        if control_value is None:
            return ResultStatus.NO_RESULT

        self._description = f"Control value = {control_value:.2f} "
        if control_value < self.error_threshold:
            self._description += (
                f"which is less than {self.error_threshold:.2f}"
            )
            return ResultStatus.BAD
        if control_value < self.warn_threshold:
            self._description += (
                f"which is less than {self.warn_threshold:.2f}"
            )
            return ResultStatus.WARN
        self._description += f"which is more than {self.warn_threshold:.2f}"
        return ResultStatus.GOOD

    @property
    def description(self) -> str:
        return self._description
