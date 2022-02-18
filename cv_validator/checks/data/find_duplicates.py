import tempfile
import urllib.request
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from scipy import spatial

from cv_validator.core.check import BaseCheck
from cv_validator.core.condition import BaseCondition, MoreThanCondition
from cv_validator.core.context import Context
from cv_validator.utils.common import check_argument
from cv_validator.utils.embedding import (
    WrapInferenceSession,
    pre_process_edge_tpu,
    supported_models,
)
from cv_validator.utils.hashing import PHash

_DUPLICATE_RATIO_THRESHOLDS = {
    "warn": 0.05,
    "error": 0.15,
}


class FindDuplicates(BaseCheck, ABC):
    def __init__(
        self,
        mode: str = "exact",
        datasource_type: str = "between",
        condition: BaseCondition = None,
    ):
        super().__init__()
        self._modes = ["exact", "approx"]
        self._datasource_types = ["train", "test", "between"]

        self.mode: str = check_argument(mode, self._modes)
        self.datasource_type: str = check_argument(
            datasource_type, self._datasource_types
        )

        if condition is None:
            self.condition = MoreThanCondition(
                warn_threshold=_DUPLICATE_RATIO_THRESHOLDS["warn"],
                error_threshold=_DUPLICATE_RATIO_THRESHOLDS["error"],
            )

    def run(self, context: Context):
        if self._datasource_types == "between":
            hash_train = self.prepare_data(context.train.params.raw)
            hash_test = self.prepare_data(context.test.params.raw)
            duplicate_pairs = self.get_duplicates(hash_train, hash_test)
            duplicate_pairs = self.filter_swapped_pairs(duplicate_pairs)
            duplicates = self.collect_duplicates(
                duplicate_pairs,
                context.train.image_paths,
                context.test.image_paths,
            )
            num_of_duplicates = sum([len(dup) for dup in duplicates.values()])
            duplicates_ratio = num_of_duplicates / len(context.test)
        else:
            if self.datasource_type == "train":
                datasource = context.train
            else:
                datasource = context.test
            hash_source = self.prepare_data(datasource.params.raw)
            duplicate_pairs = self.get_duplicates(hash_source)
            duplicate_pairs = self.filter_equal_pairs(duplicate_pairs)
            duplicate_pairs = self.filter_swapped_pairs(duplicate_pairs)
            duplicates = self.collect_duplicates(
                duplicate_pairs,
                datasource.image_paths,
            )
            num_of_duplicates = sum([len(dup) for dup in duplicates.values()])
            duplicates_ratio = num_of_duplicates / len(datasource)

        status = self.condition(duplicates_ratio)

        result_df = pd.DataFrame.from_dict(
            {
                "number of duplicates": num_of_duplicates,
                "duplicates ratio": duplicates_ratio,
                "status": status.name,
            },
            orient="index",
        )

        duplicates_df = pd.DataFrame.from_dict(
            {
                path: ", ".join(dup_paths)
                for path, dup_paths in duplicates.items()
            },
            orient="index",
        )

        self.result.update_status(status)
        self.result.add_dataset(result_df)
        self.result.add_dataset(duplicates_df)

    def prepare_data(self, all_params: List[dict]) -> np.ndarray:
        filtered_params = [params[self.param_name] for params in all_params]
        df = np.vstack(filtered_params)
        return df

    @staticmethod
    def collect_duplicates(
        duplicate_pairs: List[tuple],
        paths_left: List[Path],
        paths_right: List[Path] = None,
    ) -> Dict[Path, List[str]]:
        if paths_right is None:
            paths_right = paths_left

        result = defaultdict(list)
        for left_index, right_index in duplicate_pairs:
            left_path = paths_left[left_index].as_posix()
            right_path = paths_right[right_index].as_posix()
            result[left_path].append(right_path)
        return result

    @staticmethod
    def filter_equal_pairs(duplicate_pairs) -> List[tuple]:
        return [pair for pair in duplicate_pairs if pair[0] != pair[1]]

    @staticmethod
    def filter_swapped_pairs(duplicate_pairs) -> List[tuple]:
        return [pair for pair in duplicate_pairs if pair[0] > pair[1]]

    def get_duplicates(
        self, hash_left: np.ndarray, hash_right: np.ndarray = None
    ) -> np.ndarray:
        if hash_right is None:
            distance = self.distance_func(hash_left, hash_left)
        else:
            distance = self.distance_func(hash_left, hash_right)
        distance_mask = distance <= self.threshold
        duplicate_indices = np.where(distance_mask)
        return duplicate_indices

    @property
    @abstractmethod
    def threshold(self) -> Union[float, int]:
        pass

    @property
    @abstractmethod
    def param_name(self) -> str:
        pass

    @abstractmethod
    def distance_func(
        self, hash_left: np.ndarray, hash_right: np.ndarray
    ) -> np.ndarray:
        pass


class HashDuplicates(FindDuplicates):
    def __init__(
        self,
        mode: str = "exact",
        datasource_type: str = "between",
        condition: BaseCondition = None,
        hamming_distance_threshold: int = 10,
    ):
        super().__init__(mode, datasource_type, condition)
        self._param_name = "hash"
        if self.mode == "approx":
            self.hamming_distance_threshold = hamming_distance_threshold
        else:
            self.hamming_distance_threshold = 0
        self.phash = PHash()

    def distance_func(
        self, hash_left: np.ndarray, hash_right: np.ndarray
    ) -> np.ndarray:
        dist_func = np.vectorize(self.phash.hamming_distance)
        distance = dist_func(hash_left, hash_right.T)
        return distance

    @property
    def param_name(self) -> str:
        return self._param_name

    @property
    def threshold(self) -> Union[float, int]:
        return self.hamming_distance_threshold

    def calc_img_params(self, img: np.array) -> dict:
        result = {self.param_name: self.phash.get_hash_str(img)}
        return result

    def get_name(self) -> str:
        return "Find duplicates by phash"

    # def get_description(self) -> str:
    #     if self._datasource_types == "between":
    #         mode = "between train and test"
    #     else:
    #         mode = f"for {self._datasource_types}"
    #     return f"Find duplicates by phash {mode}"

    def get_description(self) -> str:
        return "Find duplicates by phash"


class EmbeddingDuplicates(FindDuplicates):
    def __init__(
        self,
        mode: str = "exact",
        datasource_type: str = "between",
        condition: BaseCondition = None,
        model_name: str = "efficientnet-lite4",
        model_path: str = None,
        cosine_distance_threshold: float = 0.05,
    ):
        super().__init__(mode, datasource_type, condition)
        self._param_name = "embedding"
        self.model_name = check_argument(
            model_name, list(supported_models.keys())
        )
        if model_path is None:
            tmp_dir = Path(tempfile.gettempdir())
            model_path = tmp_dir.joinpath(f"{model_name}.onnx")
        self.model_path = Path(model_path)
        self.load_model()

        self.sess = WrapInferenceSession(self.model_path.as_posix())

        if self.mode == "approx":
            self.cosine_distance_threshold = cosine_distance_threshold
        else:
            self.cosine_distance_threshold = 0

    def load_model(self):
        if self.model_path.is_file():
            return
        urllib.request.urlretrieve(
            supported_models[self.model_name], self.model_path.as_posix()
        )

    def distance_func(
        self, hash_left: np.ndarray, hash_right: np.ndarray
    ) -> np.ndarray:
        distance = spatial.distance.cdist(hash_left, hash_right, "cosine")
        return distance

    @property
    def param_name(self) -> str:
        return self._param_name

    @property
    def threshold(self) -> Union[float, int]:
        return self.cosine_distance_threshold

    def calc_img_params(self, img: np.array) -> dict:
        img_processed = pre_process_edge_tpu(img)
        img_batch = np.expand_dims(img_processed, axis=0)
        embedding = self.sess.run(None, {"images:0": img_batch})[0][0]
        result = {self.param_name: embedding}
        return result

    def get_name(self) -> str:
        return "Find duplicates by embeddings"

    # def get_description(self) -> str:
    #     if self._datasource_types == "between":
    #         mode = "between train and test"
    #     else:
    #         mode = f"for {self._datasource_types}"
    #     return f"Find duplicates by embeddings {mode}"

    def get_description(self) -> str:
        return "Find duplicates by embeddings"