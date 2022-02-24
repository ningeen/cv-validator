import numpy as np
import pandas as pd

from cv_validator.checks.data.param_distribution import ParamDistributionCheck
from cv_validator.core.data import DataSource


class ClassifierLabelDistribution(ParamDistributionCheck):
    """
    Label distribution by class

    Compares label distribution between train and test
    """

    def __init__(self):
        super().__init__()
        desired_params = ["labels"]
        self.update_desired_params(desired_params)

    def calc_img_params(self, img: np.array) -> dict:
        return dict()

    def get_source_data(self, source: DataSource) -> pd.DataFrame:
        df = pd.DataFrame({"labels": source.labels})
        return df
