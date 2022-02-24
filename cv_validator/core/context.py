from typing import Callable, List

from cv_validator.core.data import DataSource
from cv_validator.utils.common import check_class
from cv_validator.utils.metric import (
    ScorerParamsType,
    ScorerType,
    check_scorers,
    check_task,
)


class Context:
    def __init__(
        self,
        task: str,
        train: DataSource,
        test: DataSource,
        metrics: ScorerType = None,
        metrics_parameters: ScorerParamsType = None,
    ):
        self._task: str = check_task(task)
        self.train: DataSource = check_class(train, DataSource)
        self.test: DataSource = check_class(test, DataSource)
        self.metrics: List[Callable] = check_scorers(
            task, metrics, metrics_parameters
        )

    @property
    def task(self):
        return self._task
