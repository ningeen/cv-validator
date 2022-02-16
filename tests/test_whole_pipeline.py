import pytest

from cv_validator.checks.data import (
    BrightnessCheck,
    ClassifierLabelDistribution,
    ColorShift,
    ImageSize,
)
from cv_validator.core.data import DataSource
from cv_validator.core.suite import BaseSuite


def test_simple_pipeline(classification_data_all, clf_params):
    _, train_paths, train_labels, _ = classification_data_all["train"]
    _, test_paths, test_labels, _ = classification_data_all["test"]

    train = DataSource(train_paths, train_labels)
    test = DataSource(test_paths, test_labels)

    suite = BaseSuite(
        checks=[
            ImageSize(),
            ColorShift(),
            BrightnessCheck(),
            ClassifierLabelDistribution(),
        ]
    )
    suite.run(
        task="multiclass",
        train=train,
        test=test,
        num_workers=4,
    )
