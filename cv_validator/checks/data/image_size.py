from typing import List

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from ...core.check import BaseCheck
from ...core.condition import BaseCondition, MoreThanCondition
from ...core.context import Context
from ...utils.psi import calculate_psi

_DEFAULT_THRESHOLD = {
    "psi": {"warn": 0.1, "error": 0.2},
    "wasserstein_distance": {"warn": 0.5, "error": 2.0},
}
_DIFF_METRICS = {
    "psi": calculate_psi,
    "wasserstein_distance": wasserstein_distance,
}


def check_diff_metric(difference_metric: str):
    if difference_metric in _DIFF_METRICS:
        return difference_metric
    raise NotImplementedError(f"Metric {difference_metric} is not allowed")


class ImageSize(BaseCheck):
    def __init__(
        self,
        difference_metric: str = "psi",
        condition: BaseCondition = None,
    ):
        super().__init__()

        self.diff_metric = check_diff_metric(difference_metric)
        self.desired_params = ["height", "width", "ratio", "area"]
        if condition is None:
            self.condition = MoreThanCondition(
                warn_threshold=_DEFAULT_THRESHOLD[self.diff_metric]["warn"],
                error_threshold=_DEFAULT_THRESHOLD[self.diff_metric]["error"],
            )

    def get_name(self) -> str:
        return "Image size check."

    def get_description(self) -> str:
        return "Compares image sizes and ratios."

    def calc_img_params(self, img: np.array) -> dict:
        result = dict()
        result["height"] = img.shape[0]
        result["wight"] = img.shape[1]
        result["ratio"] = result["wight"] / result["height"]
        result["area"] = result["wight"] * result["height"]
        return result

    def run(self, context: Context):
        df_train = self.prepare_data(context.train.params.raw)
        df_test = self.prepare_data(context.test.params.raw)

        result = self.get_result(df_train, df_test)

        statuses = {
            param: self.condition(result[param])
            for param in self.desired_params
        }

        result_df = pd.DataFrame.from_dict(
            {
                self.diff_metric: result,
                "status": {s.name for s in statuses.values()},
            },
            orient="index",
        )

        self.result.update_status(max(statuses.values()))
        self.result.add_dataset(result_df)

    def get_result(
        self, df_train: pd.DataFrame, df_test: pd.DataFrame
    ) -> dict:
        result = {}
        diff_func = _DIFF_METRICS[self.diff_metric]
        for param in self.desired_params:
            metric = diff_func(df_train[param].values, df_test[param].values)
            result[param] = metric
        return result

    def prepare_data(self, all_params: List[dict]) -> pd.DataFrame:
        filtered_params = [self.filter_params(params) for params in all_params]
        df = pd.DataFrame(filtered_params)
        return df

    def filter_params(self, params_dict: dict) -> dict:
        filtered = {name: params_dict[name] for name in self.desired_params}
        return filtered
