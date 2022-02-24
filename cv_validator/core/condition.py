from abc import ABC, abstractmethod
from copy import deepcopy

from ..utils.common import check_class
from .status import ResultStatus


class BaseCondition(ABC):
    def __init__(self):
        self.status = ResultStatus.NO_RESULT

    def update_status(self, status: ResultStatus) -> ResultStatus:
        self.status = check_class(status, ResultStatus)
        return self.status

    @abstractmethod
    def __call__(
        self, control_value: float, param_name: str = "param"
    ) -> ResultStatus:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    def copy(self):
        return deepcopy(self)


class NoCondition(BaseCondition):
    def __call__(
        self, control_value: float, param_name: str = "param"
    ) -> ResultStatus:
        return ResultStatus.NO_RESULT

    @property
    def description(self) -> str:
        return "No condition needed"


class ThresholdCondition(BaseCondition, ABC):
    def __init__(self, warn_threshold: float, error_threshold: float):
        super().__init__()
        self.warn_threshold = warn_threshold
        self.error_threshold = error_threshold
        self._description: str = ""

    @property
    def description(self) -> str:
        return self._description

    @abstractmethod
    def __call__(
        self, control_value: float, param_name: str = "param"
    ) -> ResultStatus:
        pass


class MoreThanCondition(ThresholdCondition):
    def __call__(
        self, control_value: float, param_name: str = "param"
    ) -> ResultStatus:
        if control_value is None:
            return self.update_status(ResultStatus.NO_RESULT)

        self._description = (
            f"Control value for {param_name} = {control_value:.2f} "
        )
        if control_value > self.error_threshold:
            self._description += (
                f"which is more than {self.error_threshold:.2f}"
            )
            return self.update_status(ResultStatus.BAD)
        if control_value > self.warn_threshold:
            self._description += (
                f"which is more than {self.warn_threshold:.2f}"
            )
            return self.update_status(ResultStatus.WARN)
        self._description += f"which is less than {self.warn_threshold:.2f}"
        return self.update_status(ResultStatus.GOOD)


class LessThanCondition(ThresholdCondition):
    def __call__(
        self, control_value: float, param_name: str = "param"
    ) -> ResultStatus:
        if control_value is None:
            return self.update_status(ResultStatus.NO_RESULT)

        self._description = (
            f"Control value for {param_name} = {control_value:.2f} "
        )
        if control_value < self.error_threshold:
            self._description += (
                f"which is less than {self.error_threshold:.2f}"
            )
            return self.update_status(ResultStatus.BAD)
        if control_value < self.warn_threshold:
            self._description += (
                f"which is less than {self.warn_threshold:.2f}"
            )
            return self.update_status(ResultStatus.WARN)
        self._description += f"which is more than {self.warn_threshold:.2f}"
        return self.update_status(ResultStatus.GOD)
