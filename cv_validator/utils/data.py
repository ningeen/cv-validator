from collections.abc import Sequence
from pathlib import Path, PurePath
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

datasource_default_types = ["train", "test"]


def check_datasource_type(datasource_type: str):
    if datasource_type in datasource_default_types:
        return datasource_type
    raise TypeError(f"Unknown datasource {datasource_type}.")


def check_dir_exists(path: Union[str, Path]) -> Path:
    if isinstance(path, str):
        path = Path(path)
    if path.is_dir():
        return path
    raise FileExistsError(f"Can't find directory at {path}")


def get_image_paths(image_dir: Path):
    return [img_path for img_path in image_dir.rglob("*/*.*")]


def get_labels_from_image_paths(image_paths: List[Path]) -> Dict[str, str]:
    labels = {
        image_path.name: image_path.parent.name for image_path in image_paths
    }
    return labels


def check_labels_and_predictions(
    labels: Union[Dict, Sequence], image_names: List[str] = None
) -> Optional[Dict[str, Any]]:
    if labels is None:
        return None

    if isinstance(labels, dict):
        assert set(image_names) == set(labels.keys())
        return labels

    if (
        isinstance(labels, Sequence)
        or isinstance(labels, np.ndarray)
        or isinstance(labels, pd.Series)
    ) and not isinstance(labels, str):
        labels_dict = dict(zip(image_names, labels))
        return labels_dict

    raise TypeError(f"Unsupported labels type: {type(labels)}.")


def convert_to_path(image_paths: Sequence):
    sample = image_paths[0]

    if isinstance(image_paths[0], str):
        return [Path(path) for path in image_paths]

    if isinstance(image_paths[0], PurePath):
        return image_paths

    raise TypeError(f"Unsupported image path type: {type(sample)}.")
