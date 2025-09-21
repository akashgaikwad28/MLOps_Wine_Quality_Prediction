import pandas as pd
import numpy as np
from src.Wine_Quality_Prediction.pipeline.prediction_pipeline import PredictionPipeline

def test_prediction_pipeline_with_valid_input():
  
    feature_names = [
        "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
        "chlorides", "free sulfur dioxide", "total sulfur dioxide",
        "density", "pH", "sulphates", "alcohol"
    ]
    
    sample_input = pd.DataFrame([[
        7.4, 0.7, 0.0, 1.9, 0.076,
        11.0, 34.0, 0.9978, 3.51, 0.56, 9.4
    ]], columns=feature_names)

    pipeline = PredictionPipeline()

    # Act
    prediction = pipeline.predict(sample_input)

    # Assert
    assert prediction is not None
    assert isinstance(prediction, np.ndarray)
    assert prediction.shape[0] == 1
