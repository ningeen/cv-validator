from abc import ABC, abstractmethod

import numpy as np

from .result import CheckResult
from .status import CheckStatus


class BaseCheck(ABC):
    def __init__(self, name: str):
        self.name: str = name
        self.status: CheckStatus = CheckStatus.INITIALIZED

    @abstractmethod
    def run(self):
        pass

    def calc_img_params(self, img: np.array) -> dict:
        return dict()

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @property
    @abstractmethod
    def result(self) -> CheckResult:
        pass

    @abstractmethod
    def update_result_status(self):
        pass
