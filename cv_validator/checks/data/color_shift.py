from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
from plotly.basedatatypes import BaseFigure

from cv_validator.checks.data.param_distribution import ParamDistributionCheck
from cv_validator.core.condition import BaseCondition
from cv_validator.utils.image import Colors


class ColorShift(ParamDistributionCheck):
    """
    Image colors check

    Compares image colors distributions.
    """

    def __init__(
        self,
        condition: BaseCondition = None,
        difference_metric: str = "psi",
        need_transformed_img: bool = False,
    ):
        super().__init__(condition, difference_metric, need_transformed_img)
        self._desired_params = (
            ["num_channels", "color_mean", "color_std"]
            + [f"{color.name}_mean" for color in Colors]
            + [f"{color.name}_std" for color in Colors]
        )
        self._plot_params = ["color_mean", "color_std"]

    def calc_img_params(self, img: np.array) -> dict:
        result = dict()
        is_grey = len(img.shape) == 2 or img.shape[2] == 1

        result["num_channels"] = 1 if is_grey else img.shape[2]
        result["color_mean"] = img.mean()
        result["color_std"] = img.std()
        if not is_grey:
            for color in Colors:
                result[f"{color.name}_mean"] = np.mean(img[:, :, color.value])
                result[f"{color.name}_std"] = np.mean(img[:, :, color.value])
        return result

    @property
    def plot_params(self):
        return self._plot_params

    def get_plots(
        self, df_train: pd.DataFrame, df_test: pd.DataFrame
    ) -> List[BaseFigure]:
        plots = []
        for param in self.plot_params:
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
