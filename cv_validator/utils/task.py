compatible_tasks = [
    "binary",
    "regression",
]


def check_task(task: str) -> str:
    if task not in compatible_tasks:
        raise NotImplementedError(f"Task {task} is not supported.")
    return task
