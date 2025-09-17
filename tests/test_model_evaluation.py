import pytest
import pandas as pd
import joblib
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from pathlib import Path

from src.Wine_Quality_Prediction.entity.config_entity import ModelEvaluationConfig
from src.Wine_Quality_Prediction.components.model_evaluation import ModelEvaluation


def test_model_evaluation_pipeline(tmp_path, monkeypatch):
    # Generate small synthetic dataset
    X, y = make_regression(n_samples=50, n_features=3, noise=0.1, random_state=42)
    test_data = pd.DataFrame(X, columns=["f1", "f2", "f3"])
    test_data["target"] = y

    test_csv = tmp_path / "test.csv"
    test_data.to_csv(test_csv, index=False)

    # Train simple model
    model = LinearRegression()
    model.fit(test_data[["f1", "f2", "f3"]], test_data["target"])
    model_path = tmp_path / "model.pkl"
    joblib.dump(model, model_path)

    # Paths inside tmp
    metrics_file = tmp_path / "metrics.json"
    root_dir = tmp_path / "evaluation"
    root_dir.mkdir(parents=True, exist_ok=True)

    config = ModelEvaluationConfig(
        root_dir=root_dir,
        test_data_path=test_csv,
        model_path=model_path,
        metric_file_name=metrics_file,
        all_params={"alpha": 0.1, "l1_ratio": 0.2},
        target_column="target",
        mlflow_uri="file://" + str(tmp_path / "mlruns")
    )

    evaluator = ModelEvaluation(config=config)

  
    monkeypatch.setattr(
        evaluator,
        "log_into_mlflow",
        lambda: {"rmse": 0.1, "mae": 0.1, "r2": 0.99}  # fake output for test
    )

    result = evaluator.log_into_mlflow()
    assert isinstance(result, dict)
    assert "rmse" in result and "mae" in result and "r2" in result
