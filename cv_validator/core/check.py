from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
from plotly.basedatatypes import BaseFigure

from ..utils.check import DIFF_METRICS, DIFF_THRESHOLD, check_diff_metric
from ..utils.common import check_class
from .condition import BaseCondition, MoreThanCondition, NoCondition
from .context import Context
from .data import DataSource
from .result import CheckResult
from .status import ResultStatus

DataType = Union[np.ndarray, pd.DataFrame]


class BaseCheck(ABC):
    def __init__(self, need_transformed_img: bool = False):
        self.name: str = self.get_name()
        self.description: str = self.get_description()
        self.need_transformed_img = need_transformed_img

        self.condition: BaseCondition = NoCondition()
        self.result: CheckResult = CheckResult()

    def __repr__(self):
        return self.name

    def update_condition(self, condition: BaseCondition):
        self.condition = check_class(condition, BaseCondition)

    @abstractmethod
    def calc_img_params(self, img: np.array) -> dict:
        pass

    @abstractmethod
    def run(self, context: Context):
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
            self.condition = MoreThanCondition(
                warn_threshold=DIFF_THRESHOLD[self.diff_metric]["warn"],
                error_threshold=DIFF_THRESHOLD[self.diff_metric]["error"],
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
