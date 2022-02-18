from enum import Enum
from pathlib import Path
from typing import Callable, List

import cv2
import numpy as np
from joblib import Parallel, delayed

from cv_validator.core.check import BaseCheck


class Colors(Enum):
    RED = 0
    GREEN = 1
    BLUE = 2


def open_image(path: Path):
    path_str = path.absolute().as_posix()
    img = cv2.imread(path_str, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def run_parallel_func_on_images(
    image_paths: List[Path],
    checks: List[BaseCheck],
    transform: Callable,
    func: Callable,
    num_workers: int,
):
    result = Parallel(n_jobs=num_workers)(
        delayed(func)(path, checks, transform) for path in image_paths
    )
    return result


def apply_transform(img: np.ndarray, transform: Callable):
    try:
        # custom
        transformed_img = transform(img)
    except KeyError:
        # albumentations
        transformed_img = transform(image=img)["image"]
    return transformed_img
