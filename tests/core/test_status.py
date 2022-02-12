import pytest

from cv_validator.core.status import ResultStatus


def test_tesult_status():
    assert 5 == len(ResultStatus)
