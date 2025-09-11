import os
import pytest
from pathlib import Path
from src.Wine_Quality_Prediction.config.configuration import ConfigurationManager
from src.Wine_Quality_Prediction.components.data_ingestion import DataIngestion

def test_data_ingestion():
    # Setup config
    config_manager = ConfigurationManager()
    data_ingestion_config = config_manager.get_data_ingestion_config()
    
    # Create DataIngestion object
    data_ingestion = DataIngestion(config=data_ingestion_config)

    # Run download and extraction
    data_ingestion.download_file()
    data_ingestion.extract_zip_file()

    # Assertions
    # 1. Check if zip file was downloaded
    assert os.path.exists(data_ingestion_config.local_data_file), \
        "Zip file was not downloaded."

    # 2. Check if unzip folder exists
    assert os.path.exists(data_ingestion_config.unzip_dir), \
        "Unzip directory was not created."

    # 3. Check if at least one dataset CSV is extracted
    extracted_files = os.listdir(data_ingestion_config.unzip_dir)
    csv_files = [f for f in extracted_files if f.endswith(".csv")]
    
    assert len(csv_files) > 0, "No CSV files found in extracted data."
    assert any("winequality-red.csv" in f or "winequality-white.csv" in f for f in csv_files), \
        "Expected winequality-red.csv or winequality-white.csv not found."
