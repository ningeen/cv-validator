import pytest
from cv_validator.core.status import ResultStatus, CheckStatus


def test_tesult_status():
    assert 4 == len(ResultStatus)


def test_check_status():
    assert 2 == len(CheckStatus)
