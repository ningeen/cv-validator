from typing import Dict, List

import numpy as np
import pandas as pd

from cv_validator.core.check import BaseCheck, DataType
from cv_validator.core.condition import BaseCondition, LessThanCondition
from cv_validator.core.context import Context
from cv_validator.utils.common import check_argument
from cv_validator.utils.constants import ThresholdMetricLess


class MetricCheck(BaseCheck):
    """
    Metric quality

    Checks model quality by given metric
    """

    def __init__(
        self,
        datasource_type: str = "test",
        condition: BaseCondition = None,
    ):
        super().__init__()
        self._datasource_types = ["train", "test"]

        self.datasource_type: str = check_argument(
            datasource_type, self._datasource_types
        )

        if condition is None:
            self.condition = LessThanCondition(
                warn_threshold=ThresholdMetricLess.warn,
                error_threshold=ThresholdMetricLess.error,
            )

    def calc_img_params(self, img: np.array) -> dict:
        return dict()

    def run(self, context: Context):
        if self.datasource_type == "train":
            datasource = context.train
        else:
            datasource = context.test

        if datasource.predictions is None or datasource.labels is None:
            return

        result = dict()
        statuses = dict()
        for metric_func in context.metrics:
            metric_name = metric_func.__name__
            score = metric_func(
                datasource.labels_array, datasource.predictions_array
            )
            result[metric_name] = score

            statuses[metric_name] = self.condition(score)

        result_df = pd.DataFrame.from_dict(
            {"metric": result, "status": statuses},
            orient="index",
        )

        self.result.update_status(max(statuses.values()))
        self.result.add_dataset(result_df)

    def prepare_data(self, params: List[Dict]) -> DataType:
        pass
