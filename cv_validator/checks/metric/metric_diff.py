from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px

from cv_validator.core.check import BaseCheck, DataType
from cv_validator.core.condition import BaseCondition, MoreThanCondition
from cv_validator.core.context import Context
from cv_validator.utils.constants import ThresholdMetricDiff


class MetricDiff(BaseCheck):
    """
    Overfitting check

    Checks difference between model quality on train and test by given metric
    """

    def __init__(
        self,
        condition: BaseCondition = None,
    ):
        super().__init__()

        self.scorer_name = "metric"
        if condition is None:
            self.condition = MoreThanCondition(
                warn_threshold=ThresholdMetricDiff.warn,
                error_threshold=ThresholdMetricDiff.error,
            )

    def calc_img_params(self, img: np.array) -> dict:
        return dict()

    def run(self, context: Context):
        result = dict()
        statuses = dict()
        for metric_func in context.metrics:
            metric_name = metric_func.__name__
            result_metric = dict()
            for data_type in ["train", "test"]:
                datasource = getattr(context, data_type)
                if datasource.predictions is None or datasource.labels is None:
                    return
                result_metric[data_type] = metric_func(
                    datasource.labels_array, datasource.predictions_array
                )
            result_metric["diff abs"] = (
                result_metric["train"] - result_metric["test"]
            )
            result_metric["diff rel, %"] = (
                result_metric["diff abs"] / result_metric["train"]
            )

            status = self.condition(result_metric["diff rel, %"])
            result_metric["status"] = status.name

            result[metric_name] = result_metric
            statuses[metric_name] = status

        result_df = pd.DataFrame.from_dict(result, orient="index")

        plots = list()
        for metric_func in context.metrics:
            metric_name = metric_func.__name__
            data_types = ["train", "test"]
            values = [result[metric_name][dt] for dt in data_types]
            plot = px.bar(x=data_types, y=values, title=metric_name)
            plot.update_layout(xaxis_title=metric_name)
            plots.append(plot)

        self.result.update_status(max(statuses.values()))
        self.result.add_dataset(result_df)
        for plot in plots:
            self.result.add_plot(plot)

    def prepare_data(self, params: List[Dict]) -> DataType:
        pass
