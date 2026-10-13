import os
from us_visa.constants import *
from dataclasses import dataclass
from datetime import datetime

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")


@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)
    timestamp: str = TIMESTAMP


training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()


class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig = training_pipeline_config):
        self.data_ingestion_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME)
        self.feature_store_file_path: str = os.path.join(self.data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR, FILE_NAME)
        self.training_file_path: str = os.path.join(self.data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TRAIN_FILE_NAME)
        self.testing_file_path: str = os.path.join(self.data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TEST_FILE_NAME)
        self.train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
        self.collection_name: str = DATA_INGESTION_COLLECTION_NAME


class DataValidationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig = training_pipeline_config):
        self.data_validation_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_VALIDATION_DIR_NAME)
        self.drift_report_file_path: str = os.path.join(
            self.data_validation_dir,
            DATA_VALIDATION_DRIFT_REPORT_DIR,
            DATA_VALIDATION_DRIFT_REPORT_FILE_NAME,
        )


class DataTransformationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig = training_pipeline_config):
        self.data_transformation_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_TRANSFORMATION_DIR_NAME)
        self.transformed_train_file_path: str = os.path.join(
            self.data_transformation_dir,
            DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            TRAIN_FILE_NAME.replace("csv", "npy"),
        )
        self.transformed_test_file_path: str = os.path.join(
            self.data_transformation_dir,
            DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            TEST_FILE_NAME.replace("csv", "npy"),
        )
        self.transformed_object_file_path: str = os.path.join(
            self.data_transformation_dir,
            DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
            PREPROCSSING_OBJECT_FILE_NAME,
        )


class ModelTrainerConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig = training_pipeline_config):
        self.model_trainer_dir: str = os.path.join(training_pipeline_config.artifact_dir, MODEL_TRAINER_DIR_NAME)
        self.trained_model_file_path: str = os.path.join(
            self.model_trainer_dir,
            MODEL_TRAINER_TRAINED_MODEL_DIR,
            MODEL_FILE_NAME,
        )
        self.expected_accuracy: float = MODEL_TRAINER_EXPECTED_SCORE
        self.model_config_file_path: str = MODEL_TRAINER_MODEL_CONFIG_FILE_PATH


class ModelEvaluationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig = training_pipeline_config):
        self.changed_threshold_score: float = MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE
        self.bucket_name: str = MODEL_BUCKET_NAME
        self.s3_model_key_path: str = MODEL_FILE_NAME


class ModelPusherConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig = training_pipeline_config):
        self.bucket_name: str = MODEL_BUCKET_NAME
        self.s3_model_key_path: str = MODEL_FILE_NAME
        self.pusher_model_dir: str = os.path.join("saved_models")
        self.pusher_model_path: str = os.path.join(self.pusher_model_dir, MODEL_FILE_NAME)


class USvisaPredictorConfig:
    def __init__(self):
        self.model_file_path: str = os.path.join("saved_models", MODEL_FILE_NAME)
        self.model_bucket_name: str = MODEL_BUCKET_NAME
