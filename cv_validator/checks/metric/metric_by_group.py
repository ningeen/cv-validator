from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px

from cv_validator.core.check import BaseCheck
from cv_validator.core.condition import BaseCondition, LessThanCondition
from cv_validator.core.context import Context
from cv_validator.utils.common import check_argument
from cv_validator.utils.constants import ThresholdMetricLess


class MetricByGroup(BaseCheck, ABC):
    """
    Abstract class for metric quality grouped by parameter check
    """

    def __init__(
        self,
        datasource_type: str = "test",
        condition: BaseCondition = None,
    ):
        super().__init__()
        self._datasource_types = ["train", "test"]
        self.scorer_name = "metric"

        self.datasource_type: str = check_argument(
            datasource_type, self._datasource_types
        )

        if condition is None:
            self.condition = LessThanCondition(
                warn_threshold=ThresholdMetricLess.warn,
                error_threshold=ThresholdMetricLess.error,
            )

    @property
    @abstractmethod
    def param(self) -> str:
        pass

    @property
    @abstractmethod
    def intervals(self) -> Dict[str, pd.Interval]:
        pass

    def run(self, context: Context):
        if self.datasource_type == "train":
            datasource = context.train
        else:
            datasource = context.test

        if datasource.predictions is None or datasource.labels is None:
            return

        param = self.get_source_data(datasource)

        result = dict()
        statuses = dict()
        for group, interval in self.intervals.items():
            mask = (param > interval.left) & (param <= interval.right)

            result_interval = dict(size=sum(mask))
            status_interval = dict()
            for metric_func in context.metrics:
                metric_name = metric_func.__name__
                if np.sum(mask) > 0:
                    score = metric_func(
                        datasource.labels_array[mask],
                        datasource.predictions_array[mask],
                    )
                else:
                    score = None
                result_interval[metric_name] = score
                status_interval[metric_name] = self.condition(score)
            result_interval["status"] = max(status_interval.values()).name
            result[group] = result_interval
            statuses[group] = status_interval

        result_df = pd.DataFrame.from_dict(result, orient="index").T

        plots = list()
        for metric_func in context.metrics:
            metric_name = metric_func.__name__
            groups = list(result.keys())
            metric_result = [result[group][metric_name] for group in groups]
            plot = px.bar(
                x=groups,
                y=metric_result,
                title=metric_name,
            )
            plot.update_layout(xaxis_title=metric_name)
            plots.append(plot)

        result_status = max(max(s.values()) for s in statuses.values())
        self.result.update_status(result_status)
        self.result.add_dataset(result_df)
        for plot in plots:
            self.result.add_plot(plot)

    def prepare_data(self, params_dict: List[Dict]) -> np.ndarray:
        filtered = [params[self.param] for params in params_dict]
        return np.array(filtered)


class MetricBySize(MetricByGroup):
    """
    Metric by size

    Checks metric quality grouped by image size
    """

    def __init__(self):
        super().__init__()
        self._param = "area"
        self._intervals = {
            "small [<64x64]": pd.Interval(left=0, right=64 * 64),
            "medium [<256x256]": pd.Interval(left=64 * 64, right=256 * 256),
            "large [<1024x1024]": pd.Interval(
                left=256 * 256, right=1024 * 1024
            ),
            "x-large [>=1024x1024]": pd.Interval(
                left=1024 * 1024, right=np.inf
            ),
        }

    @property
    def param(self) -> str:
        return self._param

    @property
    def intervals(self) -> Dict[str, pd.Interval]:
        return self._intervals

    def calc_img_params(self, img: np.array) -> dict:
        result = {"area": img.shape[0] * img.shape[1]}
        return result


class MetricByRatio(MetricByGroup):
    """
    Metric by ratio

    Checks metric quality grouped by image ratio
    """

    def __init__(self):
        super().__init__()
        self._param = "ratio"
        self._intervals = {
            "narrow": pd.Interval(left=0.0, right=0.99999),
            "square": pd.Interval(left=0.99999, right=1.0),
            "wide": pd.Interval(left=1.0, right=np.inf),
        }

    @property
    def param(self) -> str:
        return self._param

    @property
    def intervals(self) -> Dict[str, pd.Interval]:
        return self._intervals

    def calc_img_params(self, img: np.array) -> dict:
        height = img.shape[0]
        width = img.shape[1]
        result = {"ratio": width / height}
        return result
