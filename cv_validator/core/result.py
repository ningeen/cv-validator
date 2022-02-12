from .status import ResultStatus


class CheckResult:
    def __init__(self):
        self.status = ResultStatus.NO_RESULT
        self.datasets: list = list()
        self.plots: list = list()

    def __repr__(self) -> str:
        return self.status.name

    def is_good(self):
        self.status = ResultStatus.GOOD

    def is_warn(self):
        self.status = ResultStatus.WARN

    def is_bad(self):
        self.status = ResultStatus.BAD
