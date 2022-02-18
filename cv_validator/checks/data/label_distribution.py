import numpy as np
import pandas as pd

from ...core.check import ParamDistributionCheck
from ...core.data import DataSource


class ClassifierLabelDistribution(ParamDistributionCheck):
    def __init__(self):
        super().__init__()
        self.desired_params = ["labels"]

    def get_name(self) -> str:
        return "Label distribution."

    def get_description(self) -> str:
        return "Compares label distribution between train and test."

    def calc_img_params(self, img: np.array) -> dict:
        return dict()

    def get_source_data(self, source: DataSource) -> pd.DataFrame:
        df = pd.DataFrame({"labels": source.labels})
        return df
