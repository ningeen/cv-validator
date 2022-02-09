import pytest
from pathlib import Path
import yaml
from dataclasses import dataclass


@dataclass(frozen=True)
class ClassifierParams:
    path_to_clf_dir: Path = Path("./samples/classification").resolve()
    classes: tuple = ("apple", "orange", "banana")
    num_classes: int = len(classes)


@pytest.fixture(scope="session")
def clf_params() -> ClassifierParams:
    return ClassifierParams()


@pytest.fixture(scope="session")
def classification_data(request):
    folder = request.param
    image_dir = ClassifierParams.path_to_clf_dir / folder
    labels_path = ClassifierParams.path_to_clf_dir / "labels.yaml"

    image_paths = [path for path in image_dir.glob("*/*.jpg")]

    with open(labels_path, "r") as f:
        labels = yaml.load(f, Loader=yaml.FullLoader)

    predictions = labels
    return image_dir, image_paths, labels, predictions
