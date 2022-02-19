from pathlib import Path
from typing import Dict, List, Mapping, Optional, Union

import numpy as np
import pandas as pd

from ...core.check import BaseCheck
from ...core.condition import BaseCondition, MoreThanCondition
from ...core.context import Context
from ...utils.data import check_datasource_type

_DEFAULT_WARN_THRESHOLD = 0.00
_DEFAULT_ERROR_THRESHOLD = 0.10


class ImageExists(BaseCheck):
    def __init__(
        self,
        condition: BaseCondition = None,
        datasource: str = "train",
    ):
        super().__init__()

        self.datasource: str = check_datasource_type(datasource)
        if condition is None:
            self.condition = MoreThanCondition(
                warn_threshold=_DEFAULT_WARN_THRESHOLD,
                error_threshold=_DEFAULT_ERROR_THRESHOLD,
            )

    def get_name(self) -> str:
        return "Image exists check."

    def get_description(self) -> str:
        return "Checks if all provided images exists."

    def calc_img_params(self, img: np.array) -> dict:
        return dict()

    def run(self, context: Context):
        result = self.get_result(context.train.image_paths)
        result_df = pd.DataFrame.from_dict(
            {self.datasource: result}, orient="index"
        )

        status = self.condition(result["Not found ratio"])
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
