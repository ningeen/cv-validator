import pytest
from pathlib import Path, PurePath
from cv_validator.utils.data import *


@pytest.mark.parametrize('classification_data', ['train'], indirect=True)
def test_check_dir_exists(classification_data):
    image_dir, _, _, _ = classification_data
    assert _check_dir_exists(image_dir)
    with pytest.raises(FileExistsError):
        _check_dir_exists(image_dir / "abracadabra")


@pytest.mark.parametrize('classification_data', ['train'], indirect=True)
def test_get_full_paths(classification_data):
    train_dir, image_paths, _, _ = classification_data
    full_paths = _get_full_paths(train_dir, image_paths)

    assert len(image_paths) == len(full_paths)
    for path in full_paths:
        assert isinstance(path, PurePath)
        assert Path(path).is_file()
        assert path.relative_to(train_dir) in image_paths


@pytest.mark.parametrize('classification_data', ['train'], indirect=True)
def test_get_image_paths(classification_data):
    train_dir, image_paths, _, _ = classification_data
    image_names = [path.as_posix() for path in image_paths]

    image_paths_result = _get_image_paths(train_dir, image_names)
    image_paths_result_wo_names = _get_image_paths(train_dir)

    assert len(image_paths_result) == len(image_paths)
    assert len(image_paths_result_wo_names) == len(image_paths)
    assert set(image_paths_result) == set(image_paths)
    assert set(image_paths_result_wo_names) == set(image_paths)


@pytest.mark.parametrize('classification_data', ['train'], indirect=True)
def test_get_labels_from_image_names(classification_data):
    train_dir, image_paths, labels, _ = classification_data
    image_names = [path.as_posix() for path in image_paths]

    image_paths_result = _get_image_paths(train_dir, image_names)
    image_paths_result_wo_names = _get_image_paths(train_dir)

    assert len(image_paths_result) == len(image_paths)
    assert len(image_paths_result_wo_names) == len(image_paths)
    assert set(image_paths_result) == set(image_paths)
    assert set(image_paths_result_wo_names) == set(image_paths)
