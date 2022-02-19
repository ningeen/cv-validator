import contextlib
from enum import Enum
from pathlib import Path
from typing import Callable, List, Optional

import cv2
import joblib
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

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


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib
    to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def run_parallel_func_on_images(
    image_paths: List[Path],
    checks: List[BaseCheck],
    transform: Optional[Callable],
    func: Callable,
    num_workers: int,
):
    total = len(image_paths)
    with tqdm_joblib(tqdm(desc="Calc image params", total=total)):
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
