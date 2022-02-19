from typing import Callable, List, Union

from ..utils.common import check_class
from ..utils.metric import check_scorers, check_task
from .data import DataSource


class Context:
    def __init__(
        self,
        task: str,
        train: DataSource,
        test: DataSource,
        model: Callable = None,
        metrics: Union[str, Callable] = None,
    ):
        self._task: str = check_task(task)
        self.train: DataSource = check_class(train, DataSource)
        self.test: DataSource = check_class(test, DataSource)
        self.model: Callable = model
        self.metrics: List[Callable] = check_scorers(metrics, task)

    @property
    def task(self):
        return self._task
