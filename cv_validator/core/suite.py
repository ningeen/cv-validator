from pathlib import Path
from typing import Callable, List

from IPython.display import display

from ..utils.image import open_image, run_parallel_func_on_images
from ..utils.metric import ScorerTypes
from .check import BaseCheck
from .context import Context
from .data import DataSource
from .status import ResultStatus


class BaseSuite:
    def __init__(self):
        self.checks: List[BaseCheck] = list()
        self._context: Context = None

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
        self.prepare_image_params(self._context.train, num_workers)
        self.prepare_image_params(self._context.test, num_workers)
        self.run_checks(skip_finished)
        self.show_result()

    def run_checks(self, skip_finished: bool):
        for check in self.checks:
            finished_check = check.result.status != ResultStatus.INITIALIZED
            if skip_finished and finished_check:
                continue
            check.run(self._context)

    def prepare_image_params(self, source: DataSource, num_workers):
        train_params = run_parallel_func_on_images(
            source.image_paths,
            source.transform,
            self.calc_params,
            num_workers,
        )
        source.update_raw_params([param["raw"] for param in train_params])
        source.update_transformed_params(
            [param["transformed"] for param in train_params]
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

    def show_result(self):
        for check in self.checks:
            print("-" * 30)
            print(check.name)
            print(check.description)
            print("Result:")
            print(check.condition.description)
            for df in check.result.datasets:
                display(df)
            for plot in check.result.plots:
                plot.show()

    def save_result(self, check_name: str):
        raise NotImplementedError

    def save_results(self):
        raise NotImplementedError
