from pathlib import Path
from typing import Callable, Dict, Iterable, List, Union

from IPython.display import Markdown, display
from tqdm import tqdm

from cv_validator.core.check import BaseCheck
from cv_validator.core.context import Context
from cv_validator.core.data import DataSource
from cv_validator.utils.check import get_name_and_description
from cv_validator.utils.display import result_to_color
from cv_validator.utils.image import (
    apply_transform,
    open_image,
    run_parallel_func_on_images,
)
from cv_validator.utils.metric import ScorerParamsType, ScorerType


class BaseSuite:
    def __init__(self, checks: Union[BaseCheck, Iterable[BaseCheck]] = None):
        self._context: Context = None

        self.checks: List[BaseCheck] = list()
        if self.checks is not None:
            if isinstance(checks, BaseCheck):
                self.add_check(checks)
            elif isinstance(checks, Iterable):
                for check in checks:
                    self.add_check(check)
            else:
                raise TypeError(f"Can't add checks of type {type(checks)}")

    def run(
        self,
        task: str,
        train: DataSource,
        test: DataSource,
        metrics: ScorerType = None,
        metrics_parameters: ScorerParamsType = None,
        num_workers: int = 1,
        skip_finished: bool = True,
    ):
        self._context = Context(task, train, test, metrics, metrics_parameters)
        self.prepare_image_params(
            self._context.train, num_workers, skip_finished
        )
        self.prepare_image_params(
            self._context.test, num_workers, skip_finished
        )
        self.run_checks(skip_finished)
        self.show_result()

    def run_checks(self, skip_finished: bool):
        assert self._context is not None
        pbar = tqdm(self.checks, desc="Running checks", total=len(self.checks))
        for check in pbar:
            check_name, _ = get_name_and_description(check)
            pbar.set_postfix_str(f"Processing {check_name}")
            if skip_finished and check.have_result:
                continue
            check.run(self._context)

    def prepare_image_params(
        self, source: DataSource, num_workers: int, skip_finished: bool
    ):
        if skip_finished:
            checks = self.unfinished_checks
        else:
            checks = self.checks

        train_params = run_parallel_func_on_images(
            source.image_paths,
            checks,
            source.transform,
            self.calc_params,
            num_workers,
        )
        source.update_raw_params([param["raw"] for param in train_params])
        source.update_transformed_params(
            [param["transformed"] for param in train_params]
        )

    @staticmethod
    def calc_params(
        img_path: Path, checks: List[BaseCheck], transform: Callable
    ) -> Dict[str, Dict[str, float]]:
        img = open_image(img_path)
        transformed_img = None
        if transform is not None:
            transformed_img = apply_transform(img, transform)

        params: Dict[str, Dict[str, float]] = {
            "raw": dict(),
            "transformed": dict(),
        }
        for check in checks:
            if check.need_transformed_img and transformed_img is None:
                check_params = check.calc_img_params(img)
                params["transformed"].update(check_params)
            elif check.need_transformed_img:
                check_params = check.calc_img_params(transformed_img)
                params["transformed"].update(check_params)
            else:
                check_params = check.calc_img_params(img)
                params["raw"].update(check_params)

        return params

    def show_result(self):
        for check in self.checks:
            check_name, check_description = get_name_and_description(check)

            color = result_to_color(check.result.status)
            conditions_result = "\n".join(
                [
                    f"<span style='color: {color}'>{condition.description}</span>**"
                    for condition in check.conditions
                ]
            )
            text = Markdown(
                f"---\n"
                f"## {check_name}\n"
                f"### {check_description}\n"
                f"**Result:\n"
                f"{conditions_result}"
            )
            display(text)
            for df in check.result.datasets:
                display(df)
            for plot in check.result.plots:
                plot.show()

    @property
    def unfinished_checks(self) -> List[BaseCheck]:
        filtered = [check for check in self.checks if not check.have_result]
        return filtered

    def add_check(self, check: BaseCheck):
        if not isinstance(check, BaseCheck):
            raise TypeError("Provided check is not inherited from BaseCheck")
        self.checks.append(check)

    def save_result(self, check_name: str):
        raise NotImplementedError

    def save_results(self):
        raise NotImplementedError
