from typing import Any


def check_class(obj: Any, desired_type: Any):
    if isinstance(obj, desired_type):
        return obj
    raise TypeError(f"Expected {desired_type} class, got: {type(obj)}")
