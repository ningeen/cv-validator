import pytest

from cv_validator.checks.data import (
    BrightnessCheck,
    ClassifierLabelDistribution,
    ColorShift,
    EmbeddingDuplicates,
    HashDuplicates,
    ImageSize,
    MetricByRatio,
    MetricBySize,
    MetricCheck,
    MetricDiff,
)
from cv_validator.core.data import DataSource
from cv_validator.core.suite import BaseSuite


def test_simple_pipeline(classification_data_all, clf_params):
    _, train_paths, train_labels, train_predictions = classification_data_all[
        "train"
    ]
    _, test_paths, test_labels, test_predictions = classification_data_all[
        "test"
    ]

    train = DataSource(train_paths, train_labels, train_predictions)
    test = DataSource(test_paths, test_labels, test_predictions)

    suite = BaseSuite(
        checks=[
            ImageSize(difference_metric="psi"),
            ImageSize(difference_metric="wasserstein_distance"),
            ColorShift(),
            BrightnessCheck(),
            ClassifierLabelDistribution(),
            HashDuplicates(mode="exact", datasource_type="train"),
            EmbeddingDuplicates(mode="approx", datasource_type="between"),
            MetricCheck(),
            MetricDiff(),
            MetricBySize(),
            MetricByRatio(),
        ]
    )
    suite.run(
        task="multiclass",
        train=train,
        test=test,
        num_workers=4,
    )
