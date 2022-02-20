<p align="center">
<a href="https://pypi.org/project/cv-validator" target="_blank">
    <img src="https://img.shields.io/pypi/v/cv-validator?color=%2334D058&label=pypi%20package" alt="Package version">
</a>
</p>

# CV validator
Library to validate computer vision data and models.

## Installation
```commandline
pip install cv-validator
```

## Usage
Example on colab: [Link](https://colab.research.google.com/drive/184BZS6iJJTtAyHMY34TOS-W-MjpiOqCW?usp=sharing)

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
