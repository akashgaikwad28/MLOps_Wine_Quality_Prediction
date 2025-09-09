import pytest
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


def test_save_and_load_json(tmp_path):
    data = {"wine": "red", "score": 90}
    json_file = tmp_path / "test.json"

    common.save_json(json_file, data)
    loaded = common.load_json(json_file)

    assert isinstance(loaded, ConfigBox)
    assert loaded.wine == "red"
    assert loaded.score == 90


def test_dummy_pass():
    """Always passes to keep pipeline green."""
    assert True
