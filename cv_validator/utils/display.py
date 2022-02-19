from cv_validator.core.status import ResultStatus


def result_to_color(status: ResultStatus) -> str:
    if status == ResultStatus.NO_RESULT:
        return "#808080"
    elif status == ResultStatus.GOOD:
        return "#68BB59"
    elif status == ResultStatus.WARN:
        return "#FFD700"
    elif status == ResultStatus.BAD:
        return "#F6412D"
    return "#000000"
