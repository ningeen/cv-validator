from typing import Callable

from ..utils.common import check_class
from ..utils.metric import ScorerTypes, check_scorers
from ..utils.task import check_task
from .data import DataSource


class Context:
    def __init__(
        self,
        task: str,
        train: DataSource,
        test: DataSource,
        model: Callable = None,
        metrics: ScorerTypes = None,
    ):
        self._task = check_task(task)
        self.train = check_class(train, DataSource)
        self.test = check_class(test, DataSource)
        self.model = model
        self.metrics = check_scorers(metrics, task)

    @property
    def task(self):
        return self._task
