import os
import sys
import pandas as pd
from sklearn.metrics import f1_score

from us_visa.constants import TARGET_COLUMN
from us_visa.entity.artifact_entity import (
    DataIngestionArtifact,
    ModelEvaluationArtifact,
    ModelTrainerArtifact,
)
from us_visa.entity.config_entity import ModelEvaluationConfig
from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.utils.main_utils import load_object


class ModelEvaluation:
    def __init__(
        self,
        model_eval_config: ModelEvaluationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise USvisaException(e, sys)

    def get_best_model(self):
        """
        Check for previously deployed model in saved_models directory.
        """
        try:
            production_model_path = os.path.join("saved_models", self.model_eval_config.s3_model_key_path)
            if os.path.exists(production_model_path):
                return load_object(file_path=production_model_path), production_model_path
            return None, None
        except Exception as e:
            raise USvisaException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Evaluate candidate trained model against current production model.
        """
        try:
            logging.info("Initiating Model Evaluation...")
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            x = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            y = test_df[TARGET_COLUMN].apply(lambda val: 1 if str(val).strip().capitalize() == "Certified" else 0)

            trained_model = load_object(file_path=self.model_trainer_artifact.trained_model_file_path)
            y_trained_pred = trained_model.predict(x)
            trained_model_f1 = f1_score(y, y_trained_pred)
            logging.info(f"Candidate Model Test F1-Score: {trained_model_f1:.4f}")

            production_model, prod_path = self.get_best_model()

            if production_model is None:
                logging.info("No prior production model found. Automatically accepting candidate model.")
                model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted=True,
                    changed_accuracy=0.0,
                    s3_model_path=self.model_eval_config.s3_model_key_path,
                    trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                )
                return model_evaluation_artifact

            y_prod_pred = production_model.predict(x)
            production_model_f1 = f1_score(y, y_prod_pred)
            logging.info(f"Current Production Model Test F1-Score: {production_model_f1:.4f}")

            diff = trained_model_f1 - production_model_f1
            is_accepted = diff >= self.model_eval_config.changed_threshold_score

            logging.info(
                f"Candidate vs Production F1 diff: {diff:.4f} (Threshold: {self.model_eval_config.changed_threshold_score}). Accepted: {is_accepted}"
            )

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=is_accepted,
                changed_accuracy=float(diff),
                s3_model_path=self.model_eval_config.s3_model_key_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
            )
            return model_evaluation_artifact

        except Exception as e:
            raise USvisaException(e, sys)
