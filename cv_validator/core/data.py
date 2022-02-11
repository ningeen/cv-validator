from abc import ABC, abstractmethod
from collections.abc import Sequence
from collections import defaultdict
from typing import List, Union, Callable, Any
from pathlib import Path, PurePath

from cv_validator.utils.data import *
from cv_validator.utils.image import *


class DataSource:
    def __init__(
        self,
        image_paths: Sequence,
        labels: Any = None,
        predictions: Any = None,
        transform: Callable = None,
        num_workers: int = 1,
    ):
        assert len(image_paths) > 0, "Empty paths"

        self.image_paths = _convert_to_path(image_paths)
        self.labels, self.class_to_labels_mapping = \
            _convert_labels_to_dict(labels, self.image_names)
        self.predictions = predictions

        self._params = ImageParams()
        self._img_hash = ImageHashParams()
        self._embeddings = EmbeddingParams()

        self.transform = transform
        self.num_workers = num_workers

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

    @property
    def params(self):
        return self._params

    @property
    def img_hash(self):
        return self._img_hash

    @property
    def embeddings(self):
        return self._embeddings

    def __iter__(self):
        return self

    def __getitem__(self, index):
        pass

    def __next__(self):
        pass

    def _get_calculate_func(
        self,
        transform: Callable = None,
        calc_stats: bool = True,
        calc_hash: bool = False,
        calc_embeddings: bool = False,
    ):
        def calc_func(path):
            result = defaultdict(None)
            img = open_image(path)
            if calc_stats:
                result['stats'], result["stats_transform"] = \
                    self._params.calculate(img, transform)
            if calc_hash:
                result['img_hash'], result["img_hash_transform"] = \
                    self._img_hash.calculate(img, transform)
            if calc_embeddings:
                result['embeddings'], result["embeddings_transform"] = \
                    self._embeddings.calculate(img, transform)
            return result
        return calc_func

    def calculate(
        self,
        image_paths: List[Path],
        transform: Callable = None,
        calc_stats: bool = True,
        calc_hash: bool = False,
        calc_embeddings: bool = False,
        num_workers: int = 1,
    ):
        calc_func = self._get_calculate_func(
            transform, calc_stats, calc_hash, calc_embeddings
        )
        result = run_parallel_func_on_images(
            image_paths, calc_func, num_workers
        )


class DataParams(ABC):
    def __init__(self):
        self.raw_params = None
        self.transformed_params = None

    @property
    def params_calculated(self):
        return self.raw_params is not None

    @property
    def transformed_params_calculated(self):
        return self.transformed_params is not None

    @abstractmethod
    def calculate(
        self,
        image_paths: List[Path],
        transform: Callable = None,
        num_workers: int = 1,
    ):
        pass


class ImageParams(DataParams):
    pass


class ImageHashParams(DataParams):
    pass


class EmbeddingParams(DataParams):
    pass
