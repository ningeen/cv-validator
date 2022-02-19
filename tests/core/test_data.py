import pytest

from cv_validator.core.data import DataSource


@pytest.mark.parametrize("classification_data", ["train"], indirect=True)
def test_datasource_init(classification_data, clf_params):
    image_dir, image_paths, labels, predictions = classification_data

    with pytest.raises(AssertionError):
        _ = DataSource([])

    source = DataSource(image_paths)
    assert source.labels is None
    assert source.predictions is None

    source = DataSource(image_paths, labels)
    assert isinstance(source.labels, dict)
    assert len(image_paths) == len(source.labels)
    assert source.predictions is None

    source = DataSource(image_paths, labels, predictions)
    assert source.predictions is not None

    source_dir = DataSource.from_directory(image_dir, predictions)
    assert source_dir.image_paths is not None
    assert source_dir.labels is not None
    # assert source_dir.labels == source.labels
    # assert source_dir.predictions == source.predictions
