import os
import shutil
import sys

from us_visa.entity.artifact_entity import ModelEvaluationArtifact, ModelPusherArtifact
from us_visa.entity.config_entity import ModelPusherConfig
from us_visa.exception import USvisaException
from us_visa.logger import logging


class ModelPusher:
    def __init__(
        self,
        model_evaluation_artifact: ModelEvaluationArtifact,
        model_pusher_config: ModelPusherConfig = ModelPusherConfig(),
    ):
        try:
            self.model_evaluation_artifact = model_evaluation_artifact
            self.model_pusher_config = model_pusher_config
        except Exception as e:
            raise USvisaException(e, sys)

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Deploy accepted model to production model directory.
        """
        try:
            logging.info("Initiating Model Pusher...")

            if not self.model_evaluation_artifact.is_model_accepted:
                logging.info("Model not accepted during evaluation. Skipping model pusher deployment.")
                return ModelPusherArtifact(
                    bucket_name=self.model_pusher_config.bucket_name,
                    s3_model_path=self.model_pusher_config.s3_model_key_path,
                )

            os.makedirs(self.model_pusher_config.pusher_model_dir, exist_ok=True)
            shutil.copy(
                src=self.model_evaluation_artifact.trained_model_path,
                dst=self.model_pusher_config.pusher_model_path,
            )
            logging.info(f"Model successfully deployed to production directory: {self.model_pusher_config.pusher_model_path}")

            model_pusher_artifact = ModelPusherArtifact(
                bucket_name=self.model_pusher_config.bucket_name,
                s3_model_path=self.model_pusher_config.pusher_model_path,
            )
            return model_pusher_artifact

        except Exception as e:
            raise USvisaException(e, sys)
