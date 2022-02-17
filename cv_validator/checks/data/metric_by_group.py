from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import pandas as pd
import plotly.express as px

from cv_validator.core.check import BaseCheck
from cv_validator.core.condition import BaseCondition, LessThanCondition
from cv_validator.core.context import Context
from cv_validator.utils.common import check_argument


class MetricByGroup(BaseCheck, ABC):
    def __init__(
        self,
        datasource_type: str = "test",
        warn_threshold: float = 0.6,
        error_threshold: float = 0.2,
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
                warn_threshold=warn_threshold,
                error_threshold=error_threshold,
            )

    def _update_scorer_name(self, name: str):
        self.scorer_name = name

    def get_name(self) -> str:
        return "Metric check by group"

    @property
    @abstractmethod
    def param(self) -> str:
        pass

    @property
    @abstractmethod
    def intervals(self) -> Dict[str, pd.Interval]:
        pass

    def filter_param(self, params_dict: dict) -> np.ndarray:
        filtered = [params[self.param] for params in params_dict]
        return np.array(filtered)

    def run(self, context: Context):
        if self.datasource_type == "train":
            datasource = context.train
        else:
            datasource = context.test

        if len(datasource.predictions) == 0 or len(datasource.labels) == 0:
            return

        scorer = context.metrics[0]._score_func
        self._update_scorer_name(scorer.__name__)

        param = self.filter_param(datasource.params.raw)
        scores = dict()
        for group, interval in self.intervals.items():
            mask = (param > interval.left) & (param <= interval.right)
            if np.sum(mask) > 0:
                score = scorer(
                    datasource.labels_pd[mask], datasource.predictions_pd[mask]
                )
            else:
                score = None
            scores[group] = score

        statuses = {
            group: self.condition(score) for group, score in scores.items()
        }
        result_df = pd.DataFrame.from_dict(
            {
                self.scorer_name: scores,
                "status": statuses,
            },
            orient="index",
        )

        plot = px.bar(
            x=list(scores.keys()),
            y=list(scores.values()),
            title=self.scorer_name,
        )

        self.result.update_status(max(statuses.values()))
        self.result.add_dataset(result_df)
        self.result.add_plot(plot)


class MetricBySize(MetricByGroup):
    def __init__(self):
        super().__init__()
        self._param = "area"
        self._intervals = {
            "small": pd.Interval(left=0, right=64 * 64),
            "medium": pd.Interval(left=64 * 64, right=256 * 256),
            "large": pd.Interval(left=256 * 256, right=1024 * 1024),
            "x-large": pd.Interval(left=1024 * 1024, right=np.inf),
        }

    def get_description(self) -> str:
        return "Checks metric grouped by image size"

    @property
    def param(self) -> str:
        return self._param

    @property
    def intervals(self) -> Dict[str, pd.Interval]:
        return self._intervals

    def calc_img_params(self, img: np.array) -> dict:
        result = {"area": img.shape[0] * img.shape[1]}
        return result
