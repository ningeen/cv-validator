from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable, Dict, List, Union

import numpy as np
import pandas as pd

from ..utils.data import (
    check_dir_exists,
    check_labels_and_predictions,
    convert_to_path,
    get_image_paths,
    get_labels_from_image_paths,
)


class DataSource:
    def __init__(
        self,
        image_paths: Sequence,
        labels: Any = None,
        predictions: Any = None,
        transform: Callable = None,
    ):
        assert len(image_paths) > 0, "Empty paths"

        self.image_paths = convert_to_path(image_paths)
        self.labels = check_labels_and_predictions(labels, self.image_names)
        self.predictions = check_labels_and_predictions(
            predictions, self.image_names
        )

        self._params = DataParams()
        self.transform = transform

    @property
    def labels_array(self):
        return np.array([self.labels[name] for name in self.image_names])

    @property
    def predictions_array(self):
        return np.array([self.predictions[name] for name in self.image_names])

    def __len__(self):
        return len(self.image_paths)

    @classmethod
    def from_directory(
        cls,
        image_dir: Union[str, Path],
        predictions: Any = None,
        transform: Callable = None,
    ):
        image_dir = check_dir_exists(image_dir)
        image_paths = get_image_paths(image_dir)
        labels = get_labels_from_image_paths(image_paths)
        return cls(image_paths, labels, predictions, transform)

    @property
    def image_names(self):
        return [img_path.name for img_path in self.image_paths]

    @property
    def params(self):
        return self._params

    def get_params(self, transformed: bool):
        if transformed:
            return self._params.transformed
        else:
            return self._params.raw

    def update_raw_params(self, new_params: List[Dict]):
        if len(new_params) > 0 and sum(len(p) for p in new_params) > 0:
            self._params.raw = new_params
        else:
            print("Provided empty params")

    def update_transformed_params(self, new_params: List[Dict]):
        if len(new_params) > 0 and sum(len(p) for p in new_params) > 0:
            self._params.transformed = new_params
        else:
            print("Provided empty params")

    def __iter__(self):
        return self

    def __getitem__(self, index):
        pass

    def __next__(self):
        pass


class DataParams:
    def __init__(self):
        self.raw = None
        self.transformed = None

    @property
    def params_calculated(self):
        return self.raw is not None

    @property
    def transformed_params_calculated(self):
        return self.transformed is not None
