import json
import os
import sys
import pandas as pd
from scipy.stats import ks_2samp

from us_visa.constants import SCHEMA_FILE_PATH
from us_visa.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from us_visa.entity.config_entity import DataValidationConfig
from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.utils.main_utils import read_yaml_file, write_yaml_file


class DataValidation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig = DataValidationConfig(),
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise USvisaException(e, sys)

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        """
        Validate whether dataframe column count matches schema definition.
        """
        try:
            expected_columns_count = len(self._schema_config["columns"])
            status = len(dataframe.columns) == expected_columns_count
            logging.info(f"Is required columns count present: {status} (Expected: {expected_columns_count}, Found: {len(dataframe.columns)})")
            return status
        except Exception as e:
            raise USvisaException(e, sys)

    def is_column_exist(self, df: pd.DataFrame) -> bool:
        """
        Validate that every expected column exists in dataframe.
        """
        try:
            missing_numerical_columns = []
            missing_categorical_columns = []

            for column in self._schema_config["numerical_columns"]:
                if column not in df.columns:
                    missing_numerical_columns.append(column)

            for column in self._schema_config["categorical_columns"]:
                if column not in df.columns:
                    missing_categorical_columns.append(column)

            if missing_numerical_columns:
                logging.info(f"Missing numerical column: {missing_numerical_columns}")

            if missing_categorical_columns:
                logging.info(f"Missing categorical column: {missing_categorical_columns}")

            return len(missing_numerical_columns) == 0 and len(missing_categorical_columns) == 0
        except Exception as e:
            raise USvisaException(e, sys)

    def detect_dataset_drift(self, reference_df: pd.DataFrame, current_df: pd.DataFrame) -> bool:
        """
        Detect dataset drift using two-sample Kolmogorov-Smirnov test for numerical features.
        Writes drift report to drift_report_file_path.
        """
        try:
            drift_report = {}
            drift_detected = False

            for column in self._schema_config["numerical_columns"]:
                ref_series = reference_df[column]
                curr_series = current_df[column]
                ks_test = ks_2samp(ref_series, curr_series)

                # p_value < 0.05 indicates distribution drift
                is_drifted = ks_test.pvalue < 0.05
                if is_drifted:
                    drift_detected = True

                drift_report[column] = {
                    "p_value": float(ks_test.pvalue),
                    "statistic": float(ks_test.statistic),
                    "drift_status": bool(is_drifted),
                }

            drift_report["overall_drift"] = drift_detected

            drift_report_file_path = self.data_validation_config.drift_report_file_path
            write_yaml_file(file_path=drift_report_file_path, content=drift_report, replace=True)
            logging.info(f"Dataset drift evaluation completed. Overall drift detected: {drift_detected}")
            logging.info(f"Drift report saved at: {drift_report_file_path}")
            return not drift_detected

        except Exception as e:
            raise USvisaException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Run all validation checks on train and test datasets.
        """
        try:
            logging.info("Initiating Data Validation...")
            train_df = pd.read_csv(self.data_ingestion_artifact.trained_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            status = self.validate_number_of_columns(dataframe=train_df)
            if not status:
                message = "Validation error: Column count mismatch in training data."
                logging.error(message)
                return DataValidationArtifact(
                    validation_status=False,
                    message=message,
                    drift_report_file_path=self.data_validation_config.drift_report_file_path,
                )

            status = self.validate_number_of_columns(dataframe=test_df)
            if not status:
                message = "Validation error: Column count mismatch in test data."
                logging.error(message)
                return DataValidationArtifact(
                    validation_status=False,
                    message=message,
                    drift_report_file_path=self.data_validation_config.drift_report_file_path,
                )

            status = self.is_column_exist(df=train_df)
            if not status:
                message = "Validation error: Missing required columns in training data."
                logging.error(message)
                return DataValidationArtifact(
                    validation_status=False,
                    message=message,
                    drift_report_file_path=self.data_validation_config.drift_report_file_path,
                )

            status = self.is_column_exist(df=test_df)
            if not status:
                message = "Validation error: Missing required columns in test data."
                logging.error(message)
                return DataValidationArtifact(
                    validation_status=False,
                    message=message,
                    drift_report_file_path=self.data_validation_config.drift_report_file_path,
                )

            self.detect_dataset_drift(reference_df=train_df, current_df=test_df)

            data_validation_artifact = DataValidationArtifact(
                validation_status=True,
                message="Data validation succeeded. No schema violation detected.",
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )
            logging.info(f"Data Validation completed: {data_validation_artifact}")
            return data_validation_artifact

        except Exception as e:
            raise USvisaException(e, sys)
