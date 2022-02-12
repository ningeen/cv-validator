from typing import List

import pandas as pd
from plotly.basedatatypes import BaseFigure

from ..utils.common import check_class
from .status import ResultStatus


class CheckResult:
    def __init__(self):
        self._status = ResultStatus.INITIALIZED
        self.datasets: List[pd.DataFrame] = list()
        self.plots: List[BaseFigure] = list()

    def __repr__(self) -> str:
        return self._status.name

    @property
    def status(self):
        return self._status

    def is_good(self):
        self._status = ResultStatus.GOOD

    def is_warn(self):
        self._status = ResultStatus.WARN

    def is_bad(self):
        self._status = ResultStatus.BAD

    def update_status(self, status: ResultStatus):
        status = check_class(status, ResultStatus)
        self._status = status

    def add_dataset(self, dataset: pd.DataFrame):
        dataset = check_class(dataset, pd.DataFrame)
        self.datasets.append(dataset)

    def add_plot(self, plot: BaseFigure):
        plot = check_class(plot, BaseFigure)
        self.plots.append(plot)
