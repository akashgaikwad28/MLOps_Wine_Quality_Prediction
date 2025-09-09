import pytest
import json
from pathlib import Path
from box import ConfigBox
from Wine_Quality_Prediction.utils import common


def test_read_yaml(tmp_path):
    yaml_content = "name: wine\nquality: high\n"
    yaml_file = tmp_path / "test.yaml"
    yaml_file.write_text(yaml_content)

    result = common.read_yaml(yaml_file)
    assert isinstance(result, ConfigBox)
    assert result.name == "wine"
    assert result.quality == "high"


def test_read_yaml_empty_file(tmp_path):
    yaml_file = tmp_path / "empty.yaml"
    yaml_file.write_text("")
    with pytest.raises(ValueError):
        common.read_yaml(yaml_file)


def test_create_directories(tmp_path):
    dir1 = tmp_path / "dir1"
    dir2 = tmp_path / "dir2"
    # Pass tuple instead of list (works with ensure_annotations)
    common.create_directories((dir1, dir2))
    assert dir1.exists() and dir2.exists()


def test_save_and_load_json(tmp_path):
    data = {"wine": "red", "score": 90}
    json_file = tmp_path / "test.json"

    common.save_json(json_file, data)
    loaded = common.load_json(json_file)

    assert isinstance(loaded, ConfigBox)
    assert loaded.wine == "red"
    assert loaded.score == 90


def test_save_and_load_bin(tmp_path):
    data = {"wine": "white", "score": 85}
    bin_file = tmp_path / "test.pkl"

    # Wrap calls to bypass ensure_annotations Any check
    (lambda d, f: common.save_bin(d, f))(data, bin_file)
    loaded = (lambda f: common.load_bin(f))(bin_file)

    assert loaded == data


def test_save_and_load_bin_list(tmp_path):
    data = [1, 2, 3, "wine"]
    bin_file = tmp_path / "list.pkl"

    (lambda d, f: common.save_bin(d, f))(data, bin_file)
    loaded = (lambda f: common.load_bin(f))(bin_file)

    assert loaded == data
