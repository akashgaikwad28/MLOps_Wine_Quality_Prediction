import os
import pytest
import numpy as np
import pandas as pd
from src.Wine_Quality_Prediction.pipeline.prediction_pipeline import PredictionPipeline

MODEL_PATH = "artifacts/model_trainer/model.joblib"

@pytest.mark.skipif(
    not os.path.exists(MODEL_PATH),
    reason="Model file not found, skipping test."
)
def test_prediction_pipeline_with_valid_input():
    # Feature names
    feature_names = [
        "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
        "chlorides", "free sulfur dioxide", "total sulfur dioxide",
        "density", "pH", "sulphates", "alcohol"
    ]

    # Sample input data
    sample_input = pd.DataFrame([[ 
        7.4, 0.7, 0.0, 1.9, 0.076,
        11.0, 34.0, 0.9978, 3.51, 0.56, 9.4
    ]], columns=feature_names)

    pipeline = PredictionPipeline()
    prediction = pipeline.predict(sample_input)

    # Basic assertions
    assert prediction is not None
    assert isinstance(prediction, np.ndarray)
    assert prediction.shape[0] == 1
