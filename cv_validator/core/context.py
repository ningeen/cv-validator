from typing import Callable, List, Any, Union

from cv_validator.core.data import DataSource
from cv_validator.utils.metric import ScorerTypes, check_scorers
from cv_validator.utils.task import check_task
from cv_validator.utils.data import check_datasource


class Context:
    def __init__(self, task: str, train_source: DataSource, test_source: DataSource, model: Callable = None, metrics: ScorerTypes = None):
        self._task = check_task(task)
        self.train_source = check_datasource(train_source)
        self.test_source = check_datasource(test_source)
        self.model = model
        self.metrics = check_scorers(metrics, task)

    @property
    def task(self):
        return self._task
