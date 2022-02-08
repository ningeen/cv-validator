from abc import ABC, abstractmethod
from typing import List, Union, Callable, Any
from pathlib import Path

from cv_validator.utils.data import *


class DataSource:
    def __init__(
            self,
            image_dir: Union[str, Path],
            image_names: List[str] = None,
            labels: Any = None,
            predictions: Any = None,
            transform: Callable = None,
            subdir_as_label: bool = False,
    ):
        self.image_dir = _check_dir_exists(image_dir)

        image_paths = _get_image_paths(image_dir, image_names)
        image_names_provided = image_names is not None
        if not image_names_provided:
            image_names = [img_path.as_posix() for img_path in image_paths]
        self.image_names = image_names

        if labels is None and subdir_as_label:
            labels = _get_labels_from_image_names(image_paths)
        elif labels is not None and image_names:
            labels = _labels_to_dict(labels, image_names, image_names_provided)
        self.labels = labels

        self.predictions = predictions

        self._params = ImageParams(transform)
        self._embeddings = EmbeddingParams(transform)

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
