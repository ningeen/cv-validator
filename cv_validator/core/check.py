from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
from plotly.basedatatypes import BaseFigure

from cv_validator.core.condition import (
    BaseCondition,
    MoreThanCondition,
    NoCondition,
)
from cv_validator.core.context import Context
from cv_validator.core.data import DataSource
from cv_validator.core.result import CheckResult
from cv_validator.core.status import ResultStatus
from cv_validator.utils.check import (
    DIFF_METRICS,
    check_diff_metric,
    get_diff_threshold,
)
from cv_validator.utils.common import check_class

DataType = Union[np.ndarray, pd.DataFrame]


class BaseCheck(ABC):
    """
    Abstract check class
    """

    def __init__(self, need_transformed_img: bool = False):
        self.need_transformed_img = need_transformed_img

        self.condition: BaseCondition = NoCondition()
        self.result: CheckResult = CheckResult()

    def update_condition(self, condition: BaseCondition):
        self.condition = check_class(condition, BaseCondition)

    @abstractmethod
    def calc_img_params(self, img: np.array) -> dict:
        pass

    @abstractmethod
    def run(self, context: Context):
        pass

    @property
    def have_result(self):
        return self.result.status != ResultStatus.INITIALIZED

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


class ParamDistributionCheck(BaseCheck, ABC):
    """
    Abstract class for checks with parameters distribution
    """

    def __init__(
        self,
        difference_metric: str = "psi",
        condition: BaseCondition = None,
        need_transformed_img: bool = False,
    ):
        super().__init__(need_transformed_img)

        self.desired_params: List[str] = list()
        self.diff_metric = check_diff_metric(difference_metric)
        if condition is None:
            threshold = get_diff_threshold(self.diff_metric)
            self.condition = MoreThanCondition(
                warn_threshold=threshold.warn,
                error_threshold=threshold.error,
            )

    def run(self, context: Context):
        df_train, df_test = self.get_data(context)

        result = self.get_result(df_train, df_test)
        plots = self.get_plots(df_train, df_test)

        statuses = {
            param: self.condition(result[param])
            for param in self.desired_params
        }

        result_df = pd.DataFrame.from_dict(
            {
                self.diff_metric: result,
                "status": {
                    param: cond_result.name
                    for param, cond_result in statuses.items()
                },
            },
            orient="index",
        )

        self.result.update_status(max(statuses.values()))
        self.result.add_dataset(result_df)
        for plot in plots:
            self.result.add_plot(plot)

    def get_result(
        self, df_train: pd.DataFrame, df_test: pd.DataFrame
    ) -> Dict:
        result = {}
        diff_func = DIFF_METRICS[self.diff_metric]
        for param in self.desired_params:
            metric = diff_func(df_train[param].values, df_test[param].values)
            result[param] = metric
        return result

    def prepare_data(self, all_params: List[Dict]) -> pd.DataFrame:
        filtered_params = [self.filter_params(params) for params in all_params]
        df = pd.DataFrame(filtered_params)
        return df

    def filter_params(self, params_dict: Dict) -> Dict:
        filtered = {name: params_dict[name] for name in self.desired_params}
        return filtered

    def get_plots(
        self, df_train: pd.DataFrame, df_test: pd.DataFrame
    ) -> List[BaseFigure]:
        plots = []
        for param in self.desired_params:
            values = pd.concat([df_train[param], df_test[param]])
            labels = ["train"] * len(df_train) + ["test"] * len(df_test)
            fig = px.histogram(
                x=values,
                color=labels,
                marginal="box",
                title=f"Distribution for feature {param}",
            )
            fig.update_layout(xaxis_title=param)
            plots.append(fig)
        return plots
