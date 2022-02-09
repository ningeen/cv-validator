from abc import ABC, abstractmethod
from typing import List, Union, Callable, Any
from pathlib import Path

from cv_validator.utils.data import *


class DataSource:
    def __init__(
        self,
        image_paths: List[str] = None,
        labels: Any = None,
        predictions: Any = None,
        transform: Callable = None,
    ):
        self.image_paths = image_paths
        self.labels, self.class_to_labels_mapping = \
            _convert_labels_to_dict(labels, self.image_names)
        self.predictions = predictions

        self._params = ImageParams(transform)
        self._embeddings = EmbeddingParams(transform)

    @classmethod
    def from_directory(
        cls,
        image_dir: Union[str, Path],
        predictions: Any = None,
        transform: Callable = None,
    ):
        image_dir = _check_dir_exists(image_dir)
        image_paths = _get_image_paths(image_dir)
        labels = _get_labels_from_image_paths(image_paths)
        return cls(image_paths, labels, predictions, transform)

    @property
    def image_names(self):
        return [img_path.name for img_path in self.image_paths]

    def __iter__(self):
        return self

    def __getitem__(self, index):
        pass

    def __next__(self):
        pass


class DataParams(ABC):
    def __init__(self, transform=None):
        self._params = None
        self._transformed_params = None
        self._transform = transform

    @property
    def params(self):
        return self._params

    @property
    def transformed_params(self):
        return self._transformed_params

    @property
    def params_calculated(self):
        return self._params is not None

    @abstractmethod
    def calculate(self):
        pass


class ImageParams(DataParams):
    pass


class EmbeddingParams(DataParams):
    pass
