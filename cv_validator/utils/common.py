from typing import Any, Dict, List, Union


def check_class(obj: Any, desired_type: Any):
    if isinstance(obj, desired_type):
        return obj
    raise TypeError(f"Expected {desired_type} class, got: {type(obj)}")


def check_argument(obj: Any, obj_list: Union[Dict, List]):
    if obj in obj_list:
        return obj
    raise NotImplementedError(
        f"Wrong argument {obj}. " f"Desired arguments: {' ,'.join(obj_list)}"
    )
