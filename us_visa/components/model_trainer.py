import os
import sys
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier

from us_visa.constants import CURRENT_YEAR
from us_visa.entity.artifact_entity import (
    ClassificationMetricArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from us_visa.entity.config_entity import ModelTrainerConfig
from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.utils.main_utils import load_numpy_array_data, load_object, save_object


class USvisaModel:
    """
    Wrapper class packaging preprocessing pipeline with the trained estimator.
    Handles automated feature engineering during inference.
    """
    def __init__(self, preprocessing_object, trained_model_object):
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def _prepare_data(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        df = dataframe.copy()
        if "yr_of_estab" in df.columns and "company_age" not in df.columns:
            df["company_age"] = CURRENT_YEAR - df["yr_of_estab"]

        drop_cols = [c for c in ["case_id", "yr_of_estab", "case_status"] if c in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)
        return df

    def predict(self, dataframe: pd.DataFrame) -> np.ndarray:
        try:
            df = self._prepare_data(dataframe)
            transformed_feature = self.preprocessing_object.transform(df)
            return self.trained_model_object.predict(transformed_feature)
        except Exception as e:
            raise USvisaException(e, sys)

    def predict_proba(self, dataframe: pd.DataFrame) -> np.ndarray:
        try:
            df = self._prepare_data(dataframe)
            transformed_feature = self.preprocessing_object.transform(df)
            return self.trained_model_object.predict_proba(transformed_feature)
        except Exception as e:
            raise USvisaException(e, sys)

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"


class ModelTrainer:
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig = ModelTrainerConfig(),
    ):
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config
        except Exception as e:
            raise USvisaException(e, sys)

    @classmethod
    def get_model_metric_artifact(cls, y_true, y_pred) -> ClassificationMetricArtifact:
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        return ClassificationMetricArtifact(
            f1_score=float(f1),
            precision_score=float(precision),
            recall_score=float(recall),
        )

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Train and evaluate candidate models, select the highest-performing model,
        package it with preprocessor pipeline, and save trained model artifact.
        """
        try:
            logging.info("Initiating Model Trainer...")

            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            train_arr = load_numpy_array_data(file_path=train_file_path)
            test_arr = load_numpy_array_data(file_path=test_file_path)

            x_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            x_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            models = {
                "RandomForest": RandomForestClassifier(
                    n_estimators=120, max_depth=12, min_samples_split=5, random_state=42
                ),
                "XGBoost": XGBClassifier(
                    n_estimators=120, learning_rate=0.08, max_depth=6, random_state=42, eval_metric="logloss"
                ),
                "CatBoost": CatBoostClassifier(
                    iterations=150, learning_rate=0.08, depth=6, verbose=0, random_state=42
                ),
            }

            model_report = {}
            trained_models = {}

            for model_name, model in models.items():
                logging.info(f"Training candidate model: {model_name}...")
                model.fit(x_train, y_train)
                y_test_pred = model.predict(x_test)
                test_f1 = f1_score(y_test, y_test_pred)
                test_acc = accuracy_score(y_test, y_test_pred)
                model_report[model_name] = test_f1
                trained_models[model_name] = model
                logging.info(f"{model_name} -> Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}")

            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model_obj = trained_models[best_model_name]

            logging.info(f"Selected Best Model: {best_model_name} with F1-Score: {best_model_score:.4f}")

            if best_model_score < self.model_trainer_config.expected_accuracy:
                raise Exception(
                    f"Best model score {best_model_score:.4f} is lower than expected score {self.model_trainer_config.expected_accuracy}."
                )

            preprocessing_obj = load_object(
                file_path=self.data_transformation_artifact.transformed_object_file_path
            )

            custom_model = USvisaModel(
                preprocessing_object=preprocessing_obj,
                trained_model_object=best_model_obj,
            )

            # Save the trained custom model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=custom_model,
            )
            logging.info(f"Saved custom USvisaModel object at: {self.model_trainer_config.trained_model_file_path}")

            y_pred_final = best_model_obj.predict(x_test)
            metric_artifact = self.get_model_metric_artifact(y_true=y_test, y_pred=y_pred_final)

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,
            )
            logging.info(f"Model Trainer completed: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise USvisaException(e, sys)
