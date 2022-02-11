from pathlib import Path
from typing import Callable, List

import cv2
from joblib import Parallel, delayed


def open_image(path: Path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def run_parallel_func_on_images(
    image_paths: List[Path],
    func: Callable,
    num_workers: int
):
    result = Parallel(n_jobs=num_workers)(
        delayed(func)(path) for path in image_paths
    )
    return result
