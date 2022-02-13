from typing import Callable, List, Union

from sklearn.metrics import SCORERS, get_scorer

ScorerTypes = [
    str,
    List[str],
    Callable,
    List[Callable],
    List[Union[str, Callable]],
]

task_default_scorers = {
    "binary": ["roc_auc"],
    "multiclass": ["f1_macro"],
    "regression": ["neg_mean_squared_error"],
}


def check_scorers(scorers: ScorerTypes, task: str) -> List[Callable]:
    if scorers is None:
        return task_default_scorers[task]

    if not (isinstance(scorers, list) or isinstance(scorers, tuple)):
        scorers = [scorers]

    processed_scorers = []
    for scorer in scorers:
        if isinstance(scorer, str) and scorer in SCORERS:
            sklearn_scorer = get_scorer(scorer)
        elif isinstance(scorer, Callable):
            sklearn_scorer = scorer
        else:
            raise NotImplementedError(f"Can't find scorer {scorer}")
        processed_scorers.append(sklearn_scorer)
    return processed_scorers


def check_task(task: str) -> str:
    if task not in task_default_scorers:
        raise NotImplementedError(f"Task {task} is not supported.")
    return task
