import pytest
import os
import pandas as pd
from pathlib import Path

from src.Wine_Quality_Prediction.config.configuration import ConfigurationManager
from src.Wine_Quality_Prediction.components.model_trainer import ModelTrainer


@pytest.mark.order(4)  
def test_model_trainer(tmp_path):
    """
    Test that the ModelTrainer:
    1. Reads train & test data
    2. Trains an ElasticNet model
    3. Saves the model file to the correct directory
    """

    config_manager = ConfigurationManager()
    model_trainer_config = config_manager.get_model_trainer_config()

    model_trainer_config.root_dir = tmp_path
    model_trainer_config.model_name = "test_model.pkl"

    trainer = ModelTrainer(config=model_trainer_config)
    trainer.train()

    model_path = os.path.join(model_trainer_config.root_dir, model_trainer_config.model_name)
    assert os.path.exists(model_path), "Model file was not created!"

    assert os.path.getsize(model_path) > 0, "Model file is empty!"

    train_data = pd.read_csv(model_trainer_config.train_data_path)
    assert model_trainer_config.target_column in train_data.columns, "Target column missing in training data!"
