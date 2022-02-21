from dataclasses import dataclass

EPS = 1e-15


@dataclass(frozen=True)
class ThresholdPSI:
    warn = 0.1
    error = 0.2


@dataclass(frozen=True)
class ThresholdWasserstein:
    warn = 2
    error = 5


@dataclass(frozen=True)
class ThresholdDuplicateRatio:
    warn = 0.05
    error = 0.15


@dataclass(frozen=True)
class ThresholdNoImageRatio:
    warn = 0.00
    error = 0.10


@dataclass(frozen=True)
class ThresholdRocAuc:
    warn = 0.55
    error = 0.60


@dataclass(frozen=True)
class ThresholdMetricLess:
    warn = 0.70
    error = 0.50


@dataclass(frozen=True)
class ThresholdMetricMore:
    warn = 0.50
    error = 0.70


@dataclass(frozen=True)
class ThresholdMetricDiff:
    warn = 0.10
    error = 0.30
