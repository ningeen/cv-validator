from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from cv_validator.core.check import BaseCheck
from cv_validator.core.condition import BaseCondition, MoreThanCondition
from cv_validator.core.context import Context
from cv_validator.utils.common import check_argument
from cv_validator.utils.embedding import (
    WrapInferenceSession,
    load_model,
    pre_process_edge_tpu,
    supported_models,
)

_SPLIT_PARAMS = dict(test_size=0.25, shuffle=True, random_state=0)
_RF_MODEL_PARAMS = dict(n_estimators=300, random_state=0)


class TrainTestShift(BaseCheck):
    def __init__(
        self,
        model_name: str = "efficientnet-lite4",
        model_path: str = None,
        warn_threshold: float = 0.55,
        error_threshold: float = 0.60,
        condition: BaseCondition = None,
        need_transformed_img: bool = False,
    ):
        super().__init__(need_transformed_img)
        self._param_name = "embedding"

        self.model_name = check_argument(
            model_name, list(supported_models.keys())
        )
        self.model_path = load_model(model_path, model_name)
        self.sess = WrapInferenceSession(self.model_path.as_posix())

        if condition is None:
            self.condition = MoreThanCondition(
                warn_threshold=warn_threshold,
                error_threshold=error_threshold,
            )

    @property
    def param_name(self) -> str:
        return self._param_name

    def get_name(self) -> str:
        return "Train test data shift"

    def get_description(self) -> str:
        return "Check difference train and test by neural net"

    def calc_img_params(self, img: np.array) -> dict:
        img_processed = pre_process_edge_tpu(img)
        img_batch = np.expand_dims(img_processed, axis=0)
        embedding = self.sess.run(None, {"images:0": img_batch})[0][0]
        result = {self.param_name: embedding}
        return result

    def run(self, context: Context):
        emb_train, emb_test = self.get_data(context)

        score = self.calc_difference_score(emb_test, emb_train)

        status = self.condition(score)
        result_df = pd.DataFrame.from_dict(
            {
                "roc_auc_score": score,
                "status": status.name,
            },
            orient="index",
        )

        self.result.update_status(status)
        self.result.add_dataset(result_df)

    @staticmethod
    def calc_difference_score(
        emb_test: np.ndarray, emb_train: np.ndarray
    ) -> float:
        emb = np.concatenate([emb_train, emb_test], axis=0)
        labels = np.concatenate(
            [np.zeros(len(emb_train)), np.ones(len(emb_test))]
        )
        x_train, x_test, y_train, y_test = train_test_split(
            emb,
            labels,
            stratify=labels,
            **_SPLIT_PARAMS,
        )
        model = RandomForestClassifier(**_RF_MODEL_PARAMS)
        model.fit(x_train, y_train)
        y_predict = model.predict_proba(x_test)[:, 1]
        score = roc_auc_score(y_test, y_predict)
        return score

    def prepare_data(self, all_params: List[Dict]) -> np.ndarray:
        filtered_params = [params[self.param_name] for params in all_params]
        df = np.vstack(filtered_params)
        return df
