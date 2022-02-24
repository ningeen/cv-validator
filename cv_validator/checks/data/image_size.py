import numpy as np

from cv_validator.checks.data.param_distribution import ParamDistributionCheck
from cv_validator.core.condition import BaseCondition


class ImageSize(ParamDistributionCheck):
    """
    Image size check

    Compares image size, area and ratio
    """

    def __init__(
        self,
        difference_metric: str = "psi",
        condition: BaseCondition = None,
    ):
        super().__init__(difference_metric, condition)
        self.desired_params = ["height", "width", "ratio", "area"]

    def calc_img_params(self, img: np.array) -> dict:
        result = dict()
        result["height"] = img.shape[0]
        result["width"] = img.shape[1]
        result["ratio"] = result["width"] / result["height"]
        result["area"] = result["width"] * result["height"]
        return result
