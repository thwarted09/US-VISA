import os
import sys
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, PowerTransformer, StandardScaler

from us_visa.constants import CURRENT_YEAR, SCHEMA_FILE_PATH, TARGET_COLUMN
from us_visa.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    DataValidationArtifact,
)
from us_visa.entity.config_entity import DataTransformationConfig
from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.utils.main_utils import drop_columns, read_yaml_file, save_numpy_array_data, save_object


class DataTransformation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_transformation_config: DataTransformationConfig = DataTransformationConfig(),
        data_validation_artifact: DataValidationArtifact = None,
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise USvisaException(e, sys)

    @classmethod
    def get_data_transformer_object(cls) -> ColumnTransformer:
        """
        Create and return ColumnTransformer preprocessor object.
        """
        try:
            logging.info("Initializing ColumnTransformer preprocessor pipeline...")
            schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)

            oh_columns = schema_config["oh_columns"]
            or_columns = schema_config["or_columns"]
            transform_columns = schema_config["transform_columns"]
            num_features = schema_config["num_features"]

            numeric_transformer = StandardScaler()
            oh_transformer = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
            ordinal_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

            transform_pipe = Pipeline(
                steps=[
                    ("power_transformer", PowerTransformer(method="yeo-johnson")),
                    ("scaler", StandardScaler()),
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("OneHotEncoder", oh_transformer, oh_columns),
                    ("OrdinalEncoder", ordinal_encoder, or_columns),
                    ("PowerTransformer", transform_pipe, transform_columns),
                    ("NumericTransformer", numeric_transformer, num_features),
                ],
                remainder="drop",
            )

            return preprocessor

        except Exception as e:
            raise USvisaException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Execute feature engineering, preprocessing transformation, and class resampling.
        """
        try:
            logging.info("Initiating Data Transformation...")

            if self.data_validation_artifact and not self.data_validation_artifact.validation_status:
                raise Exception(f"Validation failed: {self.data_validation_artifact.message}")

            train_df = pd.read_csv(self.data_ingestion_artifact.trained_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            logging.info(f"Loaded train data ({train_df.shape}) and test data ({test_df.shape})")

            # Feature Engineering: company_age
            train_df["company_age"] = CURRENT_YEAR - train_df["yr_of_estab"]
            test_df["company_age"] = CURRENT_YEAR - test_df["yr_of_estab"]

            # Drop unnecessary columns
            drop_cols = self._schema_config.get("drop_columns", ["case_id"]) + ["yr_of_estab"]
            train_df = drop_columns(df=train_df, cols=drop_cols)
            test_df = drop_columns(df=test_df, cols=drop_cols)

            # Separate inputs and target
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN].apply(
                lambda x: 1 if str(x).strip().capitalize() == "Certified" else 0
            )

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN].apply(
                lambda x: 1 if str(x).strip().capitalize() == "Certified" else 0
            )

            # Fit and transform
            preprocessor = self.get_data_transformer_object()
            logging.info("Fitting preprocessor pipeline on training data...")
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            # Resampling for class balance using SMOTEENN
            logging.info("Applying SMOTEENN resampling on training features...")
            smt = SMOTEENN(random_state=42)
            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                input_feature_train_arr, target_feature_train_df
            )
            logging.info(f"Post-resampling training shapes: {input_feature_train_final.shape}, {target_feature_train_final.shape}")

            # Concatenate features and targets
            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save transformed arrays
            save_numpy_array_data(
                file_path=self.data_transformation_config.transformed_train_file_path,
                array=train_arr,
            )
            save_numpy_array_data(
                file_path=self.data_transformation_config.transformed_test_file_path,
                array=test_arr,
            )

            # Save preprocessing object
            save_object(
                file_path=self.data_transformation_config.transformed_object_file_path,
                obj=preprocessor,
            )
            logging.info(f"Saved preprocessing object at: {self.data_transformation_config.transformed_object_file_path}")

            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )
            logging.info(f"Data Transformation completed: {data_transformation_artifact}")
            return data_transformation_artifact

        except Exception as e:
            raise USvisaException(e, sys)
