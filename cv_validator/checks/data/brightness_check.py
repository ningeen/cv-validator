from typing import List

import numpy as np

from cv_validator.checks.data.param_distribution import ParamDistributionCheck
from cv_validator.core.condition import BaseCondition


class BrightnessCheck(ParamDistributionCheck):
    """
    Image brightness check

    Compares image brightness and darkness distributions.
    """

    def __init__(
        self,
        difference_metric: str = "psi",
        conditions: List[BaseCondition] = None,
        need_transformed_img: bool = False,
        threshold_bright: int = 15,
        threshold_dark: int = 240,
    ):
        super().__init__(difference_metric, need_transformed_img, conditions)
        self.threshold_bright = threshold_bright
        self.threshold_dark = threshold_dark

        desired_params = ["bright_ratio", "dark_ratio"]
        self.update_desired_params(desired_params)

    def calc_img_params(self, img: np.array) -> dict:
        result = dict()
        is_grey = len(img.shape) == 2 or img.shape[2] == 1

        if not is_grey:
            img = img.mean(axis=-1)

        result["bright_ratio"] = np.mean(img < self.threshold_bright)
        result["dark_ratio"] = np.mean(img >= self.threshold_dark)
        return result
