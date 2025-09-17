from src.Wine_Quality_Prediction.config.configuration import ConfigurationManager
from src.Wine_Quality_Prediction.components.model_evaluation import ModelEvaluation
from src.Wine_Quality_Prediction import logger

STAGE_NAME = "Model evaluation stage"

class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass

    def initiate_model_evaluation(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(config=model_evaluation_config)
        model_evaluation.log_into_mlflow()