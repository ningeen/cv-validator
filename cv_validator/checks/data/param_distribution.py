from abc import ABC
from typing import Dict, List

import pandas as pd
from plotly import express as px
from plotly.basedatatypes import BaseFigure

from cv_validator.core.check import BaseCheckDifference
from cv_validator.core.condition import BaseCondition, MoreThanCondition
from cv_validator.core.context import Context
from cv_validator.utils.check import (
    DIFF_METRICS,
    check_diff_metric,
    get_diff_threshold,
)


class ParamDistributionCheck(BaseCheckDifference, ABC):
    """
    Abstract class for checks with parameters distribution
    """

    def __init__(
        self,
        difference_metric: str = "psi",
        need_transformed_img: bool = False,
        conditions: List[BaseCondition] = None,
    ):
        super().__init__(need_transformed_img)

        self._desired_params: List[str] = list()
        self.diff_metric = check_diff_metric(difference_metric)
        self.conditions = conditions

    def update_desired_params(self, desired_params: List[str]):
        self._desired_params = desired_params

        threshold = get_diff_threshold(self.diff_metric)
        if self.conditions is None:
            self.conditions = []
            for param in desired_params:
                condition = MoreThanCondition(
                    param,
                    warn_threshold=threshold.warn,
                    error_threshold=threshold.error,
                )
                self.conditions.append(condition)

    def run(self, context: Context):
        df_train, df_test = self.get_data(context)

        result = self.get_result(df_train, df_test)
        plots = self.get_plots(df_train, df_test)

        statuses = {
            param: self.conditions[idx](result[param])
            for idx, param in enumerate(self._desired_params)
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
        for param in self._desired_params:
            metric = diff_func(df_train[param].values, df_test[param].values)
            result[param] = metric
        return result

    def prepare_data(self, all_params: List[Dict]) -> pd.DataFrame:
        filtered_params = [self.filter_params(params) for params in all_params]
        df = pd.DataFrame(filtered_params)
        return df

    def filter_params(self, params_dict: Dict) -> Dict:
        filtered = {name: params_dict[name] for name in self._desired_params}
        return filtered

    def get_plots(
        self, df_train: pd.DataFrame, df_test: pd.DataFrame
    ) -> List[BaseFigure]:
        plots = []
        for param in self._desired_params:
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

    @property
    def plot_params(self):
        return self._desired_params
