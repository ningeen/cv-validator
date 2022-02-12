from abc import abstractmethod
from pathlib import Path
from typing import Callable, List

from ..utils.image import open_image, run_parallel_func_on_images
from ..utils.metric import ScorerTypes
from .check import BaseCheck
from .context import Context
from .data import DataSource
from .status import ResultStatus


class BaseSuite:
    def __init__(self):
        self.name: str = self.get_name()
        self.checks: List[BaseCheck] = list()
        self._context: Context = None

    @abstractmethod
    def get_name(self) -> str:
        pass

    def run(
        self,
        task: str,
        train: DataSource,
        test: DataSource,
        model: Callable = None,
        metrics: ScorerTypes = None,
        num_workers: int = 1,
        skip_finished: bool = True,
    ):
        self._context = Context(task, train, test, model, metrics)
        self.prepare_image_params(num_workers)
        self.run_checks(skip_finished)

    def run_checks(self, skip_finished: bool):
        for check in self.checks:
            finished_check = check.result.status != ResultStatus.INITIALIZED
            if skip_finished and finished_check:
                continue
            check.run(self._context)

    def prepare_image_params(self, num_workers):
        train_params = run_parallel_func_on_images(
            self._context.train.image_paths,
            self._context.train.transform,
            self.calc_params,
            num_workers,
        )
        test_params = run_parallel_func_on_images(
            self._context.test.image_paths,
            self._context.test.transform,
            self.calc_params,
            num_workers,
        )
        self._context.train.update_raw_params(
            [param["raw"] for param in train_params]
        )
        self._context.test.update_raw_params(
            [param["raw"] for param in test_params]
        )
        self._context.train.update_transformed_params(
            [param["transformed"] for param in train_params]
        )
        self._context.test.update_transformed_params(
            [param["transformed"] for param in test_params]
        )

    def calc_params(self, img_path: Path, transform: Callable):
        img = open_image(img_path)
        if transform is not None:
            transformed_img = transform(img)

        params = {"raw": dict(), "transformed": dict()}
        for check in self.checks:
            if check.need_transformed_img and transform is None:
                print(
                    f"Warning: {check.name} needs transformed image, "
                    f"but no transform function provided."
                )
                check.result.update_status(ResultStatus.NO_RESULT)
                continue

            if check.need_transformed_img:
                check_params = check.calc_img_params(transformed_img)
                params["transformed"].update(check_params)
            else:
                check_params = check.calc_img_params(img)
                params["raw"].update(check_params)

        return params

    def save_result(self, check_name: str):
        raise NotImplementedError

    def save_results(self):
        raise NotImplementedError
