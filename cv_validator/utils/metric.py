from typing import Any, Callable, Dict, List, NewType, Optional, Union

from sklearn import metrics

ScorerType = NewType(
    "ScorerType", Optional[Union[str, Callable, List[Union[str, Callable]]]]
)
ScorerParamsType = NewType(
    "ScorerParamsType", Optional[Union[Dict, List[Dict]]]
)

task_default_scorers = {
    "binary": ("roc_auc_score", None),
    "multiclass": ("f1_score", {"average": "macro"}),
    "regression": ("mean_squared_error", None),
}


class CVScorer:
    def __init__(self, scorer: Callable, params: Dict):
        self.scorer = scorer
        self.params = params

    @property
    def __name__(self):
        name = "CV scorer"
        if hasattr(self.scorer, "__name__"):
            name = self.scorer.__name__

        params_str = ""
        if isinstance(self.params, dict):
            params_str = ", ".join(
                [f"{k}={v}" for k, v in self.params.items()]
            )

        return f"{name}({params_str})"

    def __call__(self, target: Any, prediction: Any) -> float:
        result = self.scorer(target, prediction, **self.params)
        return result


def check_scorers(
    task: str,
    scorers: ScorerType,
    scorer_params: ScorerParamsType,
) -> List[Callable]:
    if scorers is None:
        scorers, scorer_params = task_default_scorers[task]

    if not (isinstance(scorers, list) or isinstance(scorers, tuple)):
        scorers = [scorers]
        scorer_params = [scorer_params]
    elif scorer_params is None:
        scorer_params = [None] * len(scorers)

    if len(scorers) != len(scorer_params):
        raise ValueError("Lengths do not match")

    processed_scorers = []
    for scorer, params in zip(scorers, scorer_params):
        if isinstance(scorer, str) and scorer in metrics.__all__:
            metric_func = getattr(metrics, scorer)
        elif callable(scorer):
            metric_func = scorer
        else:
            raise NotImplementedError(f"Can't find scorer {scorer}")

        if params is not None:
            metric_func = CVScorer(metric_func, params)

        processed_scorers.append(metric_func)
    return processed_scorers


def check_task(task: Optional[str]) -> str:
    if task not in task_default_scorers:
        raise NotImplementedError(f"Task {task} is not supported.")
    return task
