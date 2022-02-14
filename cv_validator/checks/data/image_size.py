import numpy as np

from ...core.check import ParamDistributionCheck
from ...core.condition import BaseCondition


class ImageSize(ParamDistributionCheck):
    def __init__(
        self,
        difference_metric: str = "psi",
        condition: BaseCondition = None,
    ):
        super().__init__(difference_metric, condition)
        self.desired_params = ["height", "width", "ratio", "area"]

    def get_name(self) -> str:
        return "Image size check."

    def get_description(self) -> str:
        return "Compares image sizes and ratios."

    def calc_img_params(self, img: np.array) -> dict:
        result = dict()
        result["height"] = img.shape[0]
        result["width"] = img.shape[1]
        result["ratio"] = result["width"] / result["height"]
        result["area"] = result["width"] * result["height"]
        return result
