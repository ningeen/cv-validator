from collections.abc import Sequence
from pathlib import Path, PurePath
from typing import List, Any, Union, Dict, Tuple

__all__ = [
    "_check_dir_exists",
    "_get_image_paths",
    "_get_labels_from_image_paths",
    "_convert_labels_to_dict",
    "_convert_to_path"
]


def _check_dir_exists(path: Union[str, Path]) -> Path:
    if isinstance(path, str):
        path = Path(path)
    if path.is_dir():
        return path
    raise FileExistsError(f"Can't find directory at {path}")


def _get_image_paths(image_dir: Path):
    return [img_path for img_path in image_dir.rglob("*/*.*")]


def _get_labels_from_image_paths(image_paths: List[Path]) -> Dict[str, str]:
    return {
        image_path.name: image_path.parent.name
        for image_path in image_paths
    }


def _convert_labels_to_dict(labels: Any, image_names: List[str] = None) -> \
        Tuple[Dict[str, int], Dict[int, Any]]:
    if labels is None:
        return None, None

    if isinstance(labels, dict):
        unique_labels = set(labels.values())
        num_labels = len(unique_labels)
        class_to_labels_mapping = dict(zip(range(num_labels), unique_labels))
        labels_to_class_mapping = dict(zip(unique_labels, range(num_labels)))
        labels_new = labels.copy()
        for image_name, label in labels.items():
            labels_new[image_name] = labels_to_class_mapping[label]
        return labels_new, class_to_labels_mapping

    if isinstance(labels, Sequence) and not isinstance(labels, str):
        unique_labels = set(labels)
        num_labels = len(unique_labels)
        class_to_labels_mapping = dict(zip(range(num_labels), unique_labels))
        labels_dict = dict(zip(image_names, labels))
        return labels_dict, class_to_labels_mapping

    raise TypeError(f"Unsupported labels type: {type(labels)}.")


def _convert_to_path(image_paths: Sequence):
    sample = image_paths[0]

    if isinstance(image_paths[0], str):
        return [Path(path) for path in image_paths]

    if isinstance(image_paths[0], PurePath):
        return image_paths

    raise TypeError(f"Unsupported image path type: {type(sample)}.")

