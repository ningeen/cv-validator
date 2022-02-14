import numpy as np

from ...core.condition import BaseCondition
from ...utils.image import Colors
from .image_size import ParamDistributionCheck


class ColorShift(ParamDistributionCheck):
    def __init__(
        self,
        difference_metric: str = "psi",
        condition: BaseCondition = None,
    ):
        super().__init__(difference_metric, condition)
        self.desired_params = (
            ["num_channels", "color_mean", "color_std"]
            + [f"{color.name}_mean" for color in Colors]
            + [f"{color.name}_std" for color in Colors]
        )

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
