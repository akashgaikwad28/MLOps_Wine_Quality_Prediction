import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import joblib

from src.Wine_Quality_Prediction.entity.config_entity import ModelEvaluationConfig
from src.Wine_Quality_Prediction.constants import *
from src.Wine_Quality_Prediction.utils.common import read_yaml, create_directories,save_json



import os
os.environ["MLFLOW_TRACKING_URI"]="https://dagshub.com/akashgaikwad746/MLOps_Wine_Quality_Prediction.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"]="akashgaikwad746"
os.environ["MLFLOW_TRACKING_PASSWORD"]="8213a7d6d852923d36fdd89244fe83cdd3eb73c7"



class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2
    
    def log_into_mlflow(self):
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]

        mlflow.set_tracking_uri(self.config.mlflow_uri)   # use tracking URI, not registry
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            predicted_qualities = model.predict(test_x)
            (rmse, mae, r2) = self.eval_metrics(test_y, predicted_qualities)
            
            # Save metrics locally
            scores = {"rmse": rmse, "mae": mae, "r2": r2}
            save_json(path=Path(self.config.metric_file_name), data=scores)

            # Log params + metrics
            mlflow.log_params(self.config.all_params)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)

            
            local_model_path = "artifacts/model_evaluation/model.pkl"
            joblib.dump(model, local_model_path)

            #  Log as plain artifact instead of mlflow.sklearn.log_model
            mlflow.log_artifact(local_model_path, artifact_path="model")
