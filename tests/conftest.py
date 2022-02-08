import pytest
from pathlib import Path
import yaml

PATH_TO_CLF_DIR = Path("./samples/classification")


@pytest.fixture(scope="function")
def classification_data(request):
    folder = request.param
    image_dir = PATH_TO_CLF_DIR / folder
    labels_path = PATH_TO_CLF_DIR / "labels.yaml"

    image_paths = [path.relative_to(image_dir) for path in image_dir.glob("*")]

    with open(labels_path, "r") as f:
        labels = yaml.load(f, Loader=yaml.FullLoader)

    predictions = labels
    return image_dir, image_paths, labels, predictions
