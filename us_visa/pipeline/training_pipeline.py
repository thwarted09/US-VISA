import sys
from us_visa.components.data_ingestion import DataIngestion
from us_visa.components.data_transformation import DataTransformation
from us_visa.components.data_validation import DataValidation
from us_visa.components.model_evaluation import ModelEvaluation
from us_visa.components.model_pusher import ModelPusher
from us_visa.components.model_trainer import ModelTrainer
from us_visa.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    DataValidationArtifact,
    ModelEvaluationArtifact,
    ModelPusherArtifact,
    ModelTrainerArtifact,
)
from us_visa.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    DataValidationConfig,
    ModelEvaluationConfig,
    ModelPusherConfig,
    ModelTrainerConfig,
    TrainingPipelineConfig,
)
from us_visa.exception import USvisaException
from us_visa.logger import logging


class TrainPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info(">>> STEP 1: Starting Data Ingestion Pipeline <<<")
            data_ingestion_config = DataIngestionConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
            return data_ingestion.initiate_data_ingestion()
        except Exception as e:
            raise USvisaException(e, sys)

    def start_data_validation(
        self, data_ingestion_artifact: DataIngestionArtifact
    ) -> DataValidationArtifact:
        try:
            logging.info(">>> STEP 2: Starting Data Validation Pipeline <<<")
            data_validation_config = DataValidationConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=data_validation_config,
            )
            return data_validation.initiate_data_validation()
        except Exception as e:
            raise USvisaException(e, sys)

    def start_data_transformation(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_artifact: DataValidationArtifact,
    ) -> DataTransformationArtifact:
        try:
            logging.info(">>> STEP 3: Starting Data Transformation Pipeline <<<")
            data_transformation_config = DataTransformationConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            data_transformation = DataTransformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_transformation_config=data_transformation_config,
                data_validation_artifact=data_validation_artifact,
            )
            return data_transformation.initiate_data_transformation()
        except Exception as e:
            raise USvisaException(e, sys)

    def start_model_trainer(
        self, data_transformation_artifact: DataTransformationArtifact
    ) -> ModelTrainerArtifact:
        try:
            logging.info(">>> STEP 4: Starting Model Trainer Pipeline <<<")
            model_trainer_config = ModelTrainerConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            model_trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=model_trainer_config,
            )
            return model_trainer.initiate_model_trainer()
        except Exception as e:
            raise USvisaException(e, sys)

    def start_model_evaluation(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ) -> ModelEvaluationArtifact:
        try:
            logging.info(">>> STEP 5: Starting Model Evaluation Pipeline <<<")
            model_evaluation_config = ModelEvaluationConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            model_evaluation = ModelEvaluation(
                model_eval_config=model_evaluation_config,
                data_ingestion_artifact=data_ingestion_artifact,
                model_trainer_artifact=model_trainer_artifact,
            )
            return model_evaluation.initiate_model_evaluation()
        except Exception as e:
            raise USvisaException(e, sys)

    def start_model_pusher(
        self, model_evaluation_artifact: ModelEvaluationArtifact
    ) -> ModelPusherArtifact:
        try:
            logging.info(">>> STEP 6: Starting Model Pusher Pipeline <<<")
            model_pusher_config = ModelPusherConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            model_pusher = ModelPusher(
                model_evaluation_artifact=model_evaluation_artifact,
                model_pusher_config=model_pusher_config,
            )
            return model_pusher.initiate_model_pusher()
        except Exception as e:
            raise USvisaException(e, sys)

    def run_pipeline(self) -> None:
        try:
            logging.info("==================================================")
            logging.info(">>> STARTING FULL US VISA TRAINING PIPELINE <<<")
            logging.info("==================================================")

            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact=data_ingestion_artifact
            )
            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact,
            )
            model_trainer_artifact = self.start_model_trainer(
                data_transformation_artifact=data_transformation_artifact
            )
            model_evaluation_artifact = self.start_model_evaluation(
                data_ingestion_artifact=data_ingestion_artifact,
                model_trainer_artifact=model_trainer_artifact,
            )
            model_pusher_artifact = self.start_model_pusher(
                model_evaluation_artifact=model_evaluation_artifact
            )

            logging.info("==================================================")
            logging.info(">>> US VISA TRAINING PIPELINE COMPLETED SUCCESSFULLY <<<")
            logging.info(f"Deployed Model: {model_pusher_artifact.s3_model_path}")
            logging.info("==================================================")

        except Exception as e:
            raise USvisaException(e, sys)
