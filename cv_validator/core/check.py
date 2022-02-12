from abc import ABC, abstractmethod

import numpy as np

from .condition import BaseCondition, NoCondition
from .context import Context
from .result import CheckResult
from .status import ResultStatus


class BaseCheck(ABC):
    def __init__(self, need_transformed_img: bool = False):
        self.name: str = self.get_name()
        self.description: str = self.get_description()
        self.need_transformed_img = need_transformed_img

        self.condition: BaseCondition = NoCondition()
        self.result: CheckResult = CheckResult()

    def __repr__(self):
        return self.name

    @abstractmethod
    def calc_img_params(self, img: np.array) -> dict:
        pass

    @abstractmethod
    def run(self, context: Context) -> CheckResult:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def get_description(self) -> str:
        pass

    @property
    def have_result(self):
        return self.result.status != ResultStatus.INITIALIZED
