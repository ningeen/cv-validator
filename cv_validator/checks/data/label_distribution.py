import numpy as np
import pandas as pd

from ...core.check import ParamDistributionCheck


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

    def get_data(self, context):
        df_train = pd.DataFrame({"labels": context.train.labels})
        df_test = pd.DataFrame({"labels": context.test.labels})
        return df_test, df_train
