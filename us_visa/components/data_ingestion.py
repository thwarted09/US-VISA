import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from us_visa.configuration.mongo_db_connection import MongoDBClient
from us_visa.constants import DATABASE_NAME
from us_visa.entity.artifact_entity import DataIngestionArtifact
from us_visa.entity.config_entity import DataIngestionConfig
from us_visa.exception import USvisaException
from us_visa.logger import logging


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise USvisaException(e, sys)

    def export_data_into_feature_store(self) -> pd.DataFrame:
        """
        Export data from MongoDB collection into feature store directory.
        Falls back to local Visadataset.csv if MongoDB is unavailable.
        """
        try:
            logging.info("Exporting data into feature store...")
            dataframe = None

            # Attempt 1: Fetch from MongoDB Atlas
            try:
                mongo_client = MongoDBClient(database_name=DATABASE_NAME)
                collection = mongo_client.database[self.data_ingestion_config.collection_name]
                raw_data = list(collection.find())
                if len(raw_data) > 0:
                    dataframe = pd.DataFrame(raw_data)
                    if "_id" in dataframe.columns:
                        dataframe.drop(columns=["_id"], inplace=True)
                    logging.info(f"Fetched {len(dataframe)} records from MongoDB Atlas collection: {self.data_ingestion_config.collection_name}")
            except Exception as mongo_err:
                logging.warning(f"Could not fetch from MongoDB ({mongo_err}). Attempting local dataset fallback...")

            # Attempt 2: Local CSV fallback
            if dataframe is None or len(dataframe) == 0:
                local_candidates = [
                    os.path.join("notebook", "Visadataset.csv"),
                    os.path.join("notebook", "EasyVisa.csv"),
                ]
                for path in local_candidates:
                    if os.path.exists(path):
                        logging.info(f"Loading local dataset from: {path}")
                        dataframe = pd.read_csv(path)
                        break

            if dataframe is None or len(dataframe) == 0:
                raise Exception("Failed to acquire visa data from both MongoDB and local fallback files.")

            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            logging.info(f"Saved dataset of shape {dataframe.shape} to feature store: {feature_store_file_path}")
            return dataframe

        except Exception as e:
            raise USvisaException(e, sys)

    def split_data_as_train_test(self, dataframe: pd.DataFrame) -> None:
        """
        Split feature store dataset into train and test sets.
        """
        try:
            logging.info("Splitting dataset into train and test sets...")
            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42,
            )

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)

            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)
            logging.info(f"Train set shape: {train_set.shape}, saved to: {self.data_ingestion_config.training_file_path}")
            logging.info(f"Test set shape: {test_set.shape}, saved to: {self.data_ingestion_config.testing_file_path}")

        except Exception as e:
            raise USvisaException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Initiate full data ingestion process.
        """
        try:
            logging.info("Initiating Data Ingestion...")
            dataframe = self.export_data_into_feature_store()
            self.split_data_as_train_test(dataframe=dataframe)

            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path,
            )
            logging.info(f"Data Ingestion completed: {data_ingestion_artifact}")
            return data_ingestion_artifact

        except Exception as e:
            raise USvisaException(e, sys)
