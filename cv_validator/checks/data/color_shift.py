from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
from plotly.basedatatypes import BaseFigure

from ...core.condition import BaseCondition
from ...utils.image import Colors
from .image_size import ParamDistributionCheck


class ColorShift(ParamDistributionCheck):
    def __init__(
        self,
        difference_metric: str = "psi",
        condition: BaseCondition = None,
        need_transformed_img: bool = False,
    ):
        super().__init__(difference_metric, condition, need_transformed_img)
        self.desired_params = (
            ["num_channels", "color_mean", "color_std"]
            + [f"{color.name}_mean" for color in Colors]
            + [f"{color.name}_std" for color in Colors]
        )
        self.plot_params = ["color_mean", "color_std"]

    def get_name(self) -> str:
        return "Image colors check."

    def get_description(self) -> str:
        return "Compares image colors distributions."

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
