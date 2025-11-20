import os
import sys
import pandas as pd
from google.cloud import storage
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml

logger=get_logger(__name__)

class DataIngestion:

    def __init__(self, config):
        self.config=config["data_ingestion"] 
        self.bucket_name=self.config["bucket_name"]
        self.file_names=self.config["bucket_file_names"]

        os.makedirs(RAW_DIR, exist_ok=True)

    
    def download_csv_from_GCP(self):

        try:
            logger.info("Starting data ingestion ....")

            client = storage.Client()
            bucket = client.bucket(self.bucket_name)

            for file_name in self.file_names:
                file_path= os.path.join(RAW_DIR,file_name)

                if file_name == "animelist.csv":
                    logger.info(f"downloading {file_name} ....")

                    blob=bucket.blob(file_name)
                    blob.download_to_filename(file_path)
                    data=pd.read_csv(file_path, nrows=5000000)
                    file_path_new=os.path.join(RAW_DIR,'truncated_'+file_name)
                    data.to_csv(file_path_new, index=False)

                    logger.info("large file detected, truncating it and saving another truncated file .... ")
                
                else:
                    logger.info(f"downloading {file_name} ....")

                    blob=bucket.blob(file_name)
                    blob.download_to_filename(file_path)

        except Exception as e:
            logger.error("Error while downloading data from GCP")
            raise CustomException("Failed to download data from GCP", e)
        

    def run(self):

        try:
            logger.info("INITIATING DATA INGESTION PROCESS ....")

            self.download_csv_from_GCP()

            logger.info("DATA INGESTION PROCESS COMPLETED SUCCESSFULLY ....")
        
        except Exception as e:
            logger.error("Error while initiating data ingestion process")
            raise CustomException("Failed to perform data ingestion from GCP", e)
        
        finally:
            logger.info("DATA INGESTION PROCESS CONCLUDED ....")

if __name__ == "__main__":
    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()