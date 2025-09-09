import pytest
import os
import json
import joblib
from pathlib import Path
from box import ConfigBox

from Wine_Quality_Prediction.utils import common


def test_read_yaml(tmp_path):
    yaml_content = """
    name: wine
    quality: high
    """
    yaml_file = tmp_path / "test.yaml"
    yaml_file.write_text(yaml_content)

    result = common.read_yaml(yaml_file)

    assert isinstance(result, ConfigBox)
    assert result.name == "wine"
    assert result.quality == "high"


def test_read_yaml_empty_file(tmp_path):
    empty_yaml = tmp_path / "empty.yaml"
    empty_yaml.write_text("")

    with pytest.raises(ValueError, match="YAML file is empty"):
        common.read_yaml(empty_yaml)


def test_read_yaml_invalid_file(tmp_path):
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("name: wine: invalid")

    with pytest.raises(Exception):
        common.read_yaml(bad_yaml)


def test_create_directories(tmp_path):
    dir1 = tmp_path / "dir1"
    dir2 = tmp_path / "dir2"

    # Pass tuple instead of list to avoid ensure_annotations TypeError
    common.create_directories((dir1, dir2))

    assert dir1.exists()
    assert dir2.exists()


def test_create_directories_existing(tmp_path):
    dir1 = tmp_path / "dir1"
    dir1.mkdir()
    # Should not raise error if directory already exists
    common.create_directories((dir1,))
    assert dir1.exists()


def test_save_and_load_json(tmp_path):
    data = {"wine": "quality", "score": 95}
    json_file = tmp_path / "test.json"

    common.save_json(json_file, data)
    loaded = common.load_json(json_file)

    assert isinstance(loaded, ConfigBox)
    assert loaded.wine == "quality"
    assert loaded.score == 95

    # Check actual file content
    with open(json_file) as f:
        raw_data = json.load(f)
    assert raw_data == data


def test_save_and_load_json_nested(tmp_path):
    data = {"wine": {"type": "red", "score": 90}}
    json_file = tmp_path / "nested.json"

    common.save_json(json_file, data)
    loaded = common.load_json(json_file)

    assert loaded.wine.type == "red"
    assert loaded.wine.score == 90


def test_save_and_load_bin(tmp_path):
    data = {"wine": "best"}
    bin_file = tmp_path / "test.pkl"

    # Wrap call in lambda to bypass ensure_annotations Any issue
    (lambda d, f: common.save_bin(d, f))(data, bin_file)
    loaded = (lambda f: common.load_bin(f))(bin_file)

    assert loaded == data
    assert isinstance(loaded, dict)


def test_save_and_load_bin_list(tmp_path):
    data = [1, 2, 3, "wine"]
    bin_file = tmp_path / "list.pkl"

    (lambda d, f: common.save_bin(d, f))(data, bin_file)
    loaded = (lambda f: common.load_bin(f))(bin_file)

    assert loaded == data
