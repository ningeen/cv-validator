import numpy as np

from ...core.condition import BaseCondition
from .image_size import ParamDistributionCheck


class BrightnessCheck(ParamDistributionCheck):
    def __init__(
        self,
        difference_metric: str = "psi",
        condition: BaseCondition = None,
        need_transformed_img: bool = False,
        threshold_bright: int = 15,
        threshold_dark: int = 240,
    ):
        super().__init__(difference_metric, condition, need_transformed_img)
        self.threshold_bright = threshold_bright
        self.threshold_dark = threshold_dark
        self.desired_params = ["bright_ratio", "dark_ratio"]

    def get_name(self) -> str:
        return "Image brightness check."

    def get_description(self) -> str:
        return "Compares image brightness distributions."

    def calc_img_params(self, img: np.array) -> dict:
        result = dict()
        is_grey = len(img.shape) == 2 or img.shape[2] == 1

        if not is_grey:
            img = img.mean(axis=-1)

        result["bright_ratio"] = np.mean(img < self.threshold_bright)
        result["dark_ratio"] = np.mean(img >= self.threshold_dark)
        return result
