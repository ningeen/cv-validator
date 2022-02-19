<p align="center">

[//]: # (<a href="https://github.com/ningeen/ml-validator/actions?query=workflow%3ATest" target="_blank">)

[//]: # (    <img src="https://github.com/ningeen/ml-validator/workflows/Test/badge.svg" alt="Test">)

[//]: # (</a>)

[//]: # (<a href="https://github.com/ningeen/ml-validator/actions?query=workflow%3APublish" target="_blank">)

[//]: # (    <img src="https://github.com/ningeen/ml-validator/workflows/Publish/badge.svg" alt="Publish">)

[//]: # (</a>)

[//]: # (<a href="https://codecov.io/gh/ningeen/ml-validator" target="_blank">)

[//]: # (    <img src="https://img.shields.io/codecov/c/github/ningeen/ml-validator?color=%2334D058" alt="Coverage">)

[//]: # (</a>)

<a href="https://pypi.org/project/cv-validator" target="_blank">

    <img src="https://img.shields.io/pypi/v/typer?color=%2334D058&label=pypi%20package" alt="Package version">

</a>

</p>

# CV validator
Library to validate computer vision data and models.

# Installation
```commandline
pip install cv-validator
```

#Usage
```python
from cv_validator.checks import *
from cv_validator.core.data import DataSource
from cv_validator.core.suite import BaseSuite

# Create class with data information
train = DataSource(train_image_paths, train_labels, train_predictions, transform=None)
test = DataSource(test_image_paths, test_labels, test_predictions, transform=transform)

# Create suite with different checks
suite = BaseSuite(
    checks=[
        ImageSize(),
        ColorShift(),
        BrightnessCheck(need_transformed_img=True),
        ClassifierLabelDistribution(),
        MetricCheck(),
        MetricDiff(),
        MetricBySize(),
        MetricByRatio(),
        HashDuplicates(mode="exact", datasource_type="train"),
    ]
)

# Run checks
suite.run(task="multiclass", train=train, test=test, num_workers=4)
```
