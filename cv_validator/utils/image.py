from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Callable, List

import cv2
import numpy as np
from joblib import Parallel, delayed


class Colors(Enum):
    RED = 0
    GREEN = 1
    BLUE = 2


def open_image(path: Path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def run_parallel_func_on_images(
    image_paths: List[Path],
    transform: Callable,
    func: Callable,
    num_workers: int,
):
    result = Parallel(n_jobs=num_workers)(
        delayed(func)(path, transform) for path in image_paths
    )
    return result


def calc_params(img: np.array):
    result = defaultdict(None)
    is_grey = len(img.shape) == 2 or img.shape[2] == 1

    result["height"] = img.shape[0]
    result["wight"] = img.shape[1]
    result["ratio"] = result["wight"] / result["height"]

    result["num_channels"] = 1 if is_grey else img.shape[2]
    result["color_mean"] = img.mean()
    if not is_grey:
        for color in Colors:
            result[f"{color.name}_mean"] = np.mean(img[:, :, color.value])

        for percentile in [5, 25, 50, 75, 95]:
            for color in Colors:
                name = f"{color.name}_perc{percentile:0>2}"
                result[name] = np.percentile(
                    img[:, :, color.value], percentile
                )
    return result
