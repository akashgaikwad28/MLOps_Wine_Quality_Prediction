

import os
import pandas as pd
import pytest
from src.Wine_Quality_Prediction.entity.config_entity import DataValidationConfig
from src.Wine_Quality_Prediction.components.data_validation import DataValiadtion

# ---------- FIXTURES ----------

@pytest.fixture
def sample_csv(tmp_path):
    """Creates a temporary CSV file with valid columns"""
    file_path = tmp_path / "data.csv"
    df = pd.DataFrame({
        "fixed acidity": [7.4, 7.8],
        "volatile acidity": [0.70, 0.88],
        "citric acid": [0.00, 0.04],
    })
    df.to_csv(file_path, index=False)
    return file_path

@pytest.fixture
def schema():
    """Mock schema that matches wine quality dataset columns"""
    return {
        "fixed acidity": "float",
        "volatile acidity": "float",
        "citric acid": "float",
        "residual sugar": "float",
        "chlorides": "float",
        "free sulfur dioxide": "float",
        "total sulfur dioxide": "float",
        "density": "float",
        "pH": "float",
        "sulphates": "float",
        "alcohol": "float",
        "quality": "int"
    }

@pytest.fixture
def config(tmp_path, sample_csv, schema):
    """Creates a mock DataValidationConfig"""
    status_file = tmp_path / "status.txt"
    return DataValidationConfig(
        root_dir=tmp_path,
        STATUS_FILE=status_file,
        unzip_data_dir=sample_csv,
        all_schema=schema
    )

# ---------- TEST CASES ----------

def test_validate_all_columns_pass(config):
    """Test when all columns match schema"""
    validator = DataValiadtion(config=config)
    result = validator.validate_all_columns()

    assert result is True
    with open(config.STATUS_FILE, "r") as f:
        content = f.read()
    assert "Validation status: True" in content


def test_validate_all_columns_fail(tmp_path, schema):
    """Test when columns don't match schema"""
    # Create CSV with extra column
    file_path = tmp_path / "bad_data.csv"
    df = pd.DataFrame({
        "fixed acidity": [7.4, 7.8],
        "extra_column": [1, 2],
    })
    df.to_csv(file_path, index=False)

    status_file = tmp_path / "status.txt"
    config = DataValidationConfig(
        root_dir=tmp_path,
        STATUS_FILE=status_file,
        unzip_data_dir=file_path,
        all_schema=schema
    )

    validator = DataValiadtion(config=config)
    result = validator.validate_all_columns()

    assert result is False
    with open(config.STATUS_FILE, "r") as f:
        content = f.read()
    assert "Validation status: False" in content
