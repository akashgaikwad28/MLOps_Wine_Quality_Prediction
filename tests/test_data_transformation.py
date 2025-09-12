import os
import pytest
import pandas as pd
from pathlib import Path
from src.Wine_Quality_Prediction.config.configuration import ConfigurationManager
from src.Wine_Quality_Prediction.components.data_transformation import DataTransformation


@pytest.mark.order(3)  # Run after ingestion & validation tests
def test_data_transformation():
    # Arrange - get config for transformation
    config_manager = ConfigurationManager()
    data_transformation_config = config_manager.get_data_transformation_config()

    # Create DataTransformation object
    transformer = DataTransformation(config=data_transformation_config)

    # Act - run train/test split
    transformer.train_test_splitting()

    # Assert 1 - check if train.csv and test.csv are created
    train_path = os.path.join(data_transformation_config.root_dir, "train.csv")
    test_path = os.path.join(data_transformation_config.root_dir, "test.csv")

    assert os.path.exists(train_path), "train.csv was not created."
    assert os.path.exists(test_path), "test.csv was not created."

    # Assert 2 - check if both files have data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    full_df = pd.read_csv(data_transformation_config.data_path)

    assert not train_df.empty, "train.csv is empty."
    assert not test_df.empty, "test.csv is empty."

    # Assert 3 - check if row count matches original dataset
    total_rows = len(train_df) + len(test_df)
    assert total_rows == len(full_df), "Row count mismatch after splitting."

    # Assert 4 - check column names are preserved
    assert list(train_df.columns) == list(full_df.columns), "Column mismatch in train.csv"
    assert list(test_df.columns) == list(full_df.columns), "Column mismatch in test.csv"
