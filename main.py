from src.Wine_Quality_Prediction import logger
from src.Wine_Quality_Prediction.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.Wine_Quality_Prediction.pipeline.data_validation_pipeline import DataValidationTrainingPipeline
STAGE_NAME = "Data Ingestion stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 

    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.initiate_data_ingestion()  # <-- Call the method

    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Data Validation stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataValidationTrainingPipeline()
   data_ingestion.initiate_data_validation()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e
