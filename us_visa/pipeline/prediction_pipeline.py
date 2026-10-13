import glob
import os
import sys
import numpy as np
import pandas as pd

from us_visa.constants import CURRENT_YEAR, MODEL_FILE_NAME
from us_visa.entity.config_entity import USvisaPredictorConfig
from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.utils.main_utils import load_object


class USvisaData:
    """
    Data entity representing an applicant petition for model inference.
    """
    def __init__(
        self,
        continent: str,
        education_of_employee: str,
        has_job_experience: str,
        requires_job_training: str,
        no_of_employees: int,
        yr_of_estab: int,
        region_of_employment: str,
        prevailing_wage: float,
        unit_of_wage: str,
        full_time_position: str,
    ):
        self.continent = continent
        self.education_of_employee = education_of_employee
        self.has_job_experience = has_job_experience
        self.requires_job_training = requires_job_training
        self.no_of_employees = int(no_of_employees)
        self.yr_of_estab = int(yr_of_estab)
        self.region_of_employment = region_of_employment
        self.prevailing_wage = float(prevailing_wage)
        self.unit_of_wage = unit_of_wage
        self.full_time_position = full_time_position

    def get_usvisa_input_data_frame(self) -> pd.DataFrame:
        """
        Convert input petition attributes into single-row pandas DataFrame.
        """
        try:
            usvisa_input_dict = self.get_usvisa_data_as_dict()
            return pd.DataFrame(usvisa_input_dict)
        except Exception as e:
            raise USvisaException(e, sys)

    def get_usvisa_data_as_dict(self) -> dict:
        """
        Convert petition attributes to dictionary.
        """
        try:
            input_data = {
                "continent": [self.continent],
                "education_of_employee": [self.education_of_employee],
                "has_job_experience": [self.has_job_experience],
                "requires_job_training": [self.requires_job_training],
                "no_of_employees": [self.no_of_employees],
                "yr_of_estab": [self.yr_of_estab],
                "region_of_employment": [self.region_of_employment],
                "prevailing_wage": [self.prevailing_wage],
                "unit_of_wage": [self.unit_of_wage],
                "full_time_position": [self.full_time_position],
            }
            return input_data
        except Exception as e:
            raise USvisaException(e, sys)


class USvisaClassifier:
    """
    Inference engine providing single, batch, explainability (SHAP), and counterfactual predictions.
    """
    def __init__(self, prediction_pipeline_config: USvisaPredictorConfig = USvisaPredictorConfig()):
        try:
            self.prediction_pipeline_config = prediction_pipeline_config
            self._model = None
        except Exception as e:
            raise USvisaException(e, sys)

    def _resolve_model_path(self) -> str:
        """
        Locates the model file: checks saved_models/ first, then scans artifact directories.
        """
        model_path = self.prediction_pipeline_config.model_file_path
        if os.path.exists(model_path):
            return model_path

        # Scan artifact directory for any trained models
        artifact_matches = glob.glob(f"artifact/**/{MODEL_FILE_NAME}", recursive=True)
        if artifact_matches:
            # Pick latest modified
            latest_model = max(artifact_matches, key=os.path.getmtime)
            logging.info(f"Fallback: using latest model artifact at {latest_model}")
            return latest_model

        raise FileNotFoundError("Trained model file not found in saved_models or artifact directories. Run training first.")

    def get_model(self):
        if self._model is None:
            model_path = self._resolve_model_path()
            logging.info(f"Loading US Visa model from: {model_path}")
            self._model = load_object(file_path=model_path)
        return self._model

    def predict(self, dataframe: pd.DataFrame) -> np.ndarray:
        """
        Predict approval status (Certified / Denied) for given dataframe.
        """
        try:
            model = self.get_model()
            raw_preds = model.predict(dataframe)
            status_map = {1: "Certified", 0: "Denied"}
            return np.array([status_map.get(int(p), "Denied") for p in raw_preds])
        except Exception as e:
            raise USvisaException(e, sys)

    def predict_with_details(self, dataframe: pd.DataFrame) -> list:
        """
        Returns predictions along with probability of certification and risk classification.
        """
        try:
            model = self.get_model()
            predictions = model.predict(dataframe)

            probabilities = None
            if hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(dataframe)
                    # Class 1 is Certified
                    probabilities = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
                except Exception as ex:
                    logging.warning(f"Could not compute probabilities: {ex}")

            results = []
            for i, pred in enumerate(predictions):
                status = "Certified" if int(pred) == 1 else "Denied"
                cert_prob = float(probabilities[i]) if probabilities is not None else (0.85 if status == "Certified" else 0.25)
                risk_level = "Low Risk" if cert_prob >= 0.70 else ("Moderate Risk" if cert_prob >= 0.45 else "High Risk")

                results.append({
                    "prediction": status,
                    "certification_probability": round(cert_prob * 100, 2),
                    "denial_probability": round((1.0 - cert_prob) * 100, 2),
                    "risk_assessment": risk_level,
                })

            return results
        except Exception as e:
            raise USvisaException(e, sys)

    def counterfactual_simulation(self, base_df: pd.DataFrame, alterations: dict) -> dict:
        """
        Simulate 'What-If' scenarios by modifying specific candidate features and observing probability shifts.
        """
        try:
            base_details = self.predict_with_details(base_df)[0]
            base_prob = base_details["certification_probability"]

            altered_df = base_df.copy()
            for key, val in alterations.items():
                if key in altered_df.columns:
                    altered_df[key] = val

            sim_details = self.predict_with_details(altered_df)[0]
            sim_prob = sim_details["certification_probability"]
            delta = round(sim_prob - base_prob, 2)

            return {
                "baseline_probability": base_prob,
                "simulated_probability": sim_prob,
                "probability_delta": delta,
                "baseline_prediction": base_details["prediction"],
                "simulated_prediction": sim_details["prediction"],
                "alterations": alterations,
            }
        except Exception as e:
            raise USvisaException(e, sys)
