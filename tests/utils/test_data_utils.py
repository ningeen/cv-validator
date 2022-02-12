import pytest

from cv_validator.utils.data import (
    check_dir_exists,
    convert_labels_to_dict,
    get_image_paths,
    get_labels_from_image_paths,
)


@pytest.mark.parametrize("classification_data", ["train"], indirect=True)
def test_check_dir_exists(classification_data):
    image_dir, _, _, _ = classification_data
    assert check_dir_exists(image_dir)
    with pytest.raises(FileExistsError):
        check_dir_exists(image_dir / "abracadabra")


@pytest.mark.parametrize("classification_data", ["train"], indirect=True)
def test_get_image_paths(classification_data):
    train_dir, image_paths, _, _ = classification_data

    image_paths_result = get_image_paths(train_dir)

    assert len(image_paths_result) == len(image_paths)
    assert set(image_paths_result) == set(image_paths)


@pytest.mark.parametrize("classification_data", ["train"], indirect=True)
def test_get_labels_from_image_paths(classification_data, clf_params):
    train_dir, image_paths, labels, _ = classification_data

    labels = get_labels_from_image_paths(image_paths)
    labels_unique = set(labels.values())

    assert clf_params.num_classes == len(labels_unique)

    for label in labels_unique:
        assert label in clf_params.classes


@pytest.mark.parametrize("classification_data", ["train"], indirect=True)
def test_convert_labels_to_dict_none(classification_data, clf_params):
    train_dir, image_paths, labels, _ = classification_data
    image_names = [img_path.name for img_path in image_paths]

    labels_dict, class_to_labels_mapping = convert_labels_to_dict(
        None, image_names
    )
    assert labels_dict is None
    assert class_to_labels_mapping is None


@pytest.mark.parametrize("classification_data", ["train"], indirect=True)
def test_convert_labels_to_dict_dict(classification_data, clf_params):
    train_dir, image_paths, labels, _ = classification_data
    image_names = [img_path.name for img_path in image_paths]

    labels_dict, class_to_labels_mapping = convert_labels_to_dict(
        labels, image_names
    )
    assert clf_params.num_classes == len(class_to_labels_mapping)
    assert set(clf_params.classes) == set(class_to_labels_mapping.values())
    assert set(range(len(class_to_labels_mapping))) == set(
        class_to_labels_mapping.keys()
    )


@pytest.mark.parametrize("classification_data", ["train"], indirect=True)
def test_convert_labels_to_dict_sequence(classification_data, clf_params):
    train_dir, image_paths, labels, _ = classification_data
    image_names = [img_path.name for img_path in image_paths]

    labels_dict, class_to_labels_mapping = convert_labels_to_dict(
        list(labels.values()), image_names
    )
    assert clf_params.num_classes == len(class_to_labels_mapping)
    assert set(clf_params.classes) == set(class_to_labels_mapping.values())
    assert set(range(len(class_to_labels_mapping))) == set(
        class_to_labels_mapping.keys()
    )


@pytest.mark.parametrize("classification_data", ["train"], indirect=True)
def test_convert_labels_to_dict_error(classification_data, clf_params):
    train_dir, image_paths, labels, _ = classification_data
    image_names = [img_path.name for img_path in image_paths]

    with pytest.raises(TypeError):
        _, _ = convert_labels_to_dict(123, image_names)

    with pytest.raises(TypeError):
        _, _ = convert_labels_to_dict("abracadabra", image_names)
