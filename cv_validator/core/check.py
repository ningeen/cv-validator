from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd

from cv_validator.core.condition import BaseCondition, NoCondition
from cv_validator.core.context import Context
from cv_validator.core.data import DataSource
from cv_validator.core.result import CheckResult
from cv_validator.core.status import ResultStatus
from cv_validator.utils.common import check_class

DataType = Union[np.ndarray, pd.DataFrame]
ConditionsType = Union[BaseCondition, List[BaseCondition]]


class BaseCheck(ABC):
    """
    Abstract check class
    """

    def __init__(
        self,
        condition: BaseCondition = None,
        need_transformed_img: bool = False,
    ):
        self.need_transformed_img = need_transformed_img
        self.result: CheckResult = CheckResult()
        if condition is None:
            self.condition = self.get_default_condition()
        else:
            self.condition = check_class(condition, BaseCondition)
        self.conditions: List[BaseCondition] = self.reset_conditions()

    def reset_conditions(self, count: int = 1) -> List[BaseCondition]:
        return [self.condition.copy() for _ in range(count)]

    def get_default_condition(self) -> BaseCondition:
        return NoCondition()

    @abstractmethod
    def calc_img_params(self, img: np.array) -> dict:
        pass

    @abstractmethod
    def run(self, context: Context):
        pass

    @property
    def have_result(self):
        return self.result.status != ResultStatus.INITIALIZED


class BaseCheckDifference(BaseCheck, ABC):
    """
    Abstract class for check between train and test
    """

    def get_data(self, context: Context) -> Tuple[DataType, DataType]:
        df_train = self.get_source_data(context.train)
        df_test = self.get_source_data(context.test)
        return df_train, df_test

    def get_source_data(self, source: DataSource) -> DataType:
        params = source.get_params(self.need_transformed_img)
        df = self.prepare_data(params)
        return df

    @abstractmethod
    def prepare_data(self, params: List[Dict]) -> DataType:
        pass
