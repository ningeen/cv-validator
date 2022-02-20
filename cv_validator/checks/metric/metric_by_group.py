from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px

from cv_validator.core.check import BaseCheck
from cv_validator.core.condition import BaseCondition, LessThanCondition
from cv_validator.core.context import Context
from cv_validator.utils.common import check_argument
from cv_validator.utils.metric import get_metric_function


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

    def run(self, context: Context):
        if self.datasource_type == "train":
            datasource = context.train
        else:
            datasource = context.test

        if datasource.predictions is None or datasource.labels is None:
            return

        scorer = get_metric_function(context.metrics[0])
        self._update_scorer_name(scorer.__name__)

        param = self.get_source_data(datasource)

        result: Dict[str, Dict[str, float]] = defaultdict(dict)
        for group, interval in self.intervals.items():
            mask = (param > interval.left) & (param <= interval.right)
            if np.sum(mask) > 0:
                score = scorer(
                    datasource.labels_array[mask],
                    datasource.predictions_array[mask],
                )
            else:
                score = None
            result["scores"][group] = score
            result["size"][group] = sum(mask)

        statuses = {
            group: self.condition(score)
            for group, score in result["scores"].items()
        }
        result_df = pd.DataFrame.from_dict(
            {
                self.scorer_name: result["scores"],
                "count": result["size"],
                "status": {
                    group: cond_result.name
                    for group, cond_result in statuses.items()
                },
            },
            orient="index",
        )

        plot = px.bar(
            x=list(result["scores"].keys()),
            y=list(result["scores"].values()),
            title=self.scorer_name,
        )

        self.result.update_status(max(statuses.values()))
        self.result.add_dataset(result_df)
        self.result.add_plot(plot)

    def prepare_data(self, params_dict: List[Dict]) -> np.ndarray:
        filtered = [params[self.param] for params in params_dict]
        return np.array(filtered)


class MetricBySize(MetricByGroup):
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


class MetricByRatio(MetricByGroup):
    def __init__(self):
        super().__init__()
        self._param = "ratio"
        self._intervals = {
            "narrow": pd.Interval(left=0.0, right=0.99999),
            "square": pd.Interval(left=0.99999, right=1.0),
            "wide": pd.Interval(left=1.0, right=np.inf),
        }

    def get_description(self) -> str:
        return "Checks metric grouped by image side ratio"

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
