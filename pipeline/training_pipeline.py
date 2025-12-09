from utils.common_functions import read_yaml
from config.paths_config import *
from src.data_processing import DataProcessor
from src.model_training import ModelTraining
from src.logger import get_logger

if __name__=="__main__":

    logger = get_logger(__name__)

    logger.info("Starting Data Processing and Model Training Pipeline....")

    # data_processor = DataProcessor(ANIMELIST_CSV,PROCESSED_DIR)
    # data_processor.run()

    model_trainer = ModelTraining(PROCESSED_DIR)
    model_trainer.train_model()

    logger.info("Data Processing and Model Training Pipeline Executed Successfully....")

