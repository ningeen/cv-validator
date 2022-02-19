from typing import Any, Callable, List, Optional, Union

from sklearn.metrics import SCORERS, get_scorer
from sklearn.metrics._scorer import _BaseScorer

task_default_scorers = {
    "binary": ["roc_auc"],
    "multiclass": ["accuracy"],
    "regression": ["neg_mean_squared_error"],
}


def check_scorers(scorers: Union[str, Callable], task: str) -> List[Callable]:
    if scorers is None:
        scorers = task_default_scorers[task]

    if not (isinstance(scorers, list) or isinstance(scorers, tuple)):
        scorers = [scorers]

    processed_scorers = []
    for scorer in scorers:
        if isinstance(scorer, str) and scorer in SCORERS:
            sklearn_scorer = get_scorer(scorer)
        elif callable(scorer):
            sklearn_scorer = scorer
        else:
            raise NotImplementedError(f"Can't find scorer {scorer}")
        processed_scorers.append(sklearn_scorer)
    return processed_scorers


def check_task(task: Optional[str]) -> str:
    if task not in task_default_scorers:
        raise NotImplementedError(f"Task {task} is not supported.")
    return task


def get_metric_function(metric: Callable) -> Callable:
    if isinstance(metric, _BaseScorer):
        return metric._score_func
    return metric
