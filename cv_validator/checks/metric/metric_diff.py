from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px

from cv_validator.core.check import BaseCheck, DataType
from cv_validator.core.condition import BaseCondition, MoreThanCondition
from cv_validator.core.context import Context
from cv_validator.utils.metric import get_metric_function


class MetricDiff(BaseCheck):
    def __init__(
        self,
        warn_threshold: float = 0.1,
        error_threshold: float = 0.3,
        condition: BaseCondition = None,
    ):
        super().__init__()

        self.scorer_name = "metric"
        if condition is None:
            self.condition = MoreThanCondition(
                warn_threshold=warn_threshold,
                error_threshold=error_threshold,
            )

    def _update_scorer_name(self, name: str):
        self.scorer_name = name

    def get_name(self) -> str:
        return "Metric difference"

    def get_description(self) -> str:
        return "Train-test metric difference"

    def calc_img_params(self, img: np.array) -> dict:
        return dict()

    def run(self, context: Context):
        scorer = get_metric_function(context.metrics[0])
        self._update_scorer_name(scorer.__name__)

        for datasource in [context.train, context.test]:
            if datasource.predictions is None or datasource.labels is None:
                return

        train_score = scorer(
            context.train.labels_array, context.train.predictions_array
        )
        test_score = scorer(
            context.test.labels_array, context.test.predictions_array
        )
        diff = train_score - test_score
        relative_diff = diff / train_score

        status = self.condition(relative_diff)
        result_df = pd.DataFrame.from_dict(
            {
                "train": {self.scorer_name: train_score},
                "test": {self.scorer_name: test_score},
                "difference": {self.scorer_name: diff},
                "relative difference": {self.scorer_name: relative_diff},
                "status": {self.scorer_name: status.name},
            },
            orient="index",
        )

        plot = px.bar(
            x=["train", "test"],
            y=[train_score, test_score],
            title=self.scorer_name,
        )

        self.result.update_status(status)
        self.result.add_dataset(result_df)
        self.result.add_plot(plot)

    def prepare_data(self, params: List[Dict]) -> DataType:
        pass
