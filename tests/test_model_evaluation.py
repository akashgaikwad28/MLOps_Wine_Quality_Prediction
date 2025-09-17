import os
import pandas as pd
import joblib
import tempfile
import json
import mlflow

from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

from src.Wine_Quality_Prediction.components.model_evaluation import ModelEvaluation
from src.Wine_Quality_Prediction.entity.config_entity import ModelEvaluationConfig


def test_model_evaluation_pipeline(tmp_path):
   
    X, y = make_regression(n_samples=100, n_features=3, noise=0.1, random_state=42)
    test_data = pd.DataFrame(X, columns=["f1", "f2", "f3"])
    test_data["target"] = y

    test_csv = tmp_path / "test.csv"
    test_data.to_csv(test_csv, index=False)

   
    model = LinearRegression()
    model.fit(test_data[["f1", "f2", "f3"]], test_data["target"])
    model_path = tmp_path / "model.pkl"
    joblib.dump(model, model_path)

   
    metrics_file = tmp_path / "metrics.json"
    root_dir = tmp_path / "evaluation"

    config = ModelEvaluationConfig(
        root_dir=root_dir,
        test_data_path=test_csv,
        model_path=model_path,
        metric_file_name=metrics_file,
        all_params={"alpha": 0.1, "l1_ratio": 0.2},
        target_column="target",
        mlflow_uri="file://" + str(tmp_path / "mlruns")   # local MLflow store
    )

  
    evaluator = ModelEvaluation(config=config)
    evaluator.log_into_mlflow()


    assert metrics_file.exists(), "Metrics JSON file was not created"
    
    with open(metrics_file, "r") as f:
        metrics = json.load(f)

    assert "rmse" in metrics
    assert "mae" in metrics
    assert "r2" in metrics

    assert isinstance(metrics["rmse"], float)
    assert isinstance(metrics["mae"], float)
    assert isinstance(metrics["r2"], float)

    
    mlruns_dir = tmp_path / "mlruns"
    assert mlruns_dir.exists(), "MLflow tracking directory was not created"
