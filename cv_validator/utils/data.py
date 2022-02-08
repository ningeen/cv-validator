from collections.abc import Sequence
from pathlib import Path, PurePath
from typing import List, Any, Union

__all__ = [
    "_check_dir_exists",
    "_get_full_paths",
    "_get_image_paths",
    "_get_labels_from_image_names",
    "_labels_to_dict",
]


def _check_dir_exists(path: Union[str, Path]) -> Path:
    if isinstance(path, str):
        path = Path(path)
    if path.is_dir():
        return path
    raise FileExistsError(f"Can't find directory at {path}")


def _get_full_paths(image_dir: Path, image_paths: List[Path]):
    return [PurePath(image_dir, img_name) for img_name in image_paths]


def _get_image_paths(image_dir: Path, image_names: List[str] = None):
    if image_names is None:
        image_paths = [
            img_name.relative_to(image_dir)
            for img_name in image_dir.rglob("*")
        ]
    else:
        image_paths = [Path(img_name) for img_name in image_names]
    return image_paths


def _get_labels_from_image_names(image_paths: List[Path]):
    return {
        image_path.name: image_path.parent.as_posix()
        for image_path in image_paths
    }


def _labels_to_dict(
        labels: Any, image_names: List[str], image_names_provided: bool
):
    if isinstance(labels, dict):
        return labels

    if image_names_provided and isinstance(labels, Sequence):
        return dict(zip(image_names, labels))

    raise ValueError("Ambiguity: Labels provided without image names")
