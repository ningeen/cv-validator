from pathlib import Path
from typing import List, Mapping, Union

import numpy as np
import pandas as pd

from cv_validator.core.check import BaseCheck
from cv_validator.core.condition import BaseCondition, MoreThanCondition
from cv_validator.core.context import Context
from cv_validator.utils.constants import ThresholdNoImageRatio
from cv_validator.utils.data import check_datasource_type


class ImageExists(BaseCheck):
    """
    Image existence check

    Checks if all provided images exists
    """

    def __init__(
        self,
        datasource_type: str = "train",
        condition: BaseCondition = None,
    ):
        super().__init__(condition)

        self.datasource: str = check_datasource_type(datasource_type)

    def get_default_condition(self):
        condition = MoreThanCondition(
            warn_threshold=ThresholdNoImageRatio.warn,
            error_threshold=ThresholdNoImageRatio.error,
        )
        return condition

    def calc_img_params(self, img: np.array) -> dict:
        return dict()

    def run(self, context: Context):
        result = self.get_result(context.train.image_paths)
        result_df = pd.DataFrame.from_dict(
            {self.datasource: result}, orient="index"
        )

        status = self.conditions[0](result["Not found ratio"])
        self.result.update_status(status)
        self.result.add_dataset(result_df)

    def get_result(
        self, image_paths: List[Path]
    ) -> Mapping[str, Union[int, float]]:
        file_exists_result = self.check_file_exists(image_paths)
        length = len(image_paths)
        result = dict()
        result["Total"] = length
        result["Not found"] = length - sum(file_exists_result)
        result["Not found ratio"] = result["Not found"] / result["Total"]
        return result

    @staticmethod
    def check_file_exists(image_paths: List[Path]) -> List[bool]:
        file_exists = list()
        for img_path in image_paths:
            is_file = img_path.is_file()
            file_exists.append(is_file)
        return file_exists
