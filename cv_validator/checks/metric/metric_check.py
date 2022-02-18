from typing import Dict, List

import numpy as np
import pandas as pd

from cv_validator.core.check import BaseCheck, DataType
from cv_validator.core.condition import BaseCondition, LessThanCondition
from cv_validator.core.context import Context
from cv_validator.utils.common import check_argument


class MetricCheck(BaseCheck):
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
        return "Metric check"

    def get_description(self) -> str:
        return "Checks model quality for metric"

    def calc_img_params(self, img: np.array) -> dict:
        return dict()

    def run(self, context: Context):
        if self.datasource_type == "train":
            datasource = context.train
        else:
            datasource = context.test

        if len(datasource.predictions) == 0 or len(datasource.labels) == 0:
            return

        scorer = context.metrics[0]._score_func
        self._update_scorer_name(scorer.__name__)

        score = scorer(datasource.labels_pd, datasource.predictions_pd)

        status = self.condition(score)
        result_df = pd.DataFrame.from_dict(
            {
                scorer.__name__: score,
                "status": status.name,
            },
            orient="index",
        )

        self.result.update_status(status)
        self.result.add_dataset(result_df)

    def prepare_data(self, params: List[Dict]) -> DataType:
        pass
