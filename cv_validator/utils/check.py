from typing import Any, Tuple

from scipy.stats import wasserstein_distance

from cv_validator.utils.psi import calculate_psi

DIFF_THRESHOLD = {
    "psi": {"warn": 0.1, "error": 0.2},
    "wasserstein_distance": {"warn": 0.5, "error": 2.0},
}
DIFF_METRICS = {
    "psi": calculate_psi,
    "wasserstein_distance": wasserstein_distance,
}


def check_diff_metric(difference_metric: str) -> str:
    if difference_metric in DIFF_METRICS:
        return difference_metric
    raise NotImplementedError(f"Metric {difference_metric} is not allowed")


def get_name_and_description(obj: Any) -> Tuple[str, str]:
    if hasattr(obj, "__doc__"):
        doc_line = obj.__doc__
        name, *description = [x.strip() for x in doc_line.strip().split("\n")]
        description = " ".join(description)
        description = description.strip()
    else:
        name = type(obj).__name__
        description = ""
    return name, description
