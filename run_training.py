from us_visa.pipeline.training_pipeline import TrainPipeline
from us_visa.logger import logging

if __name__ == "__main__":
    try:
        logging.info("Starting US Visa training pipeline execution...")
        pipeline = TrainPipeline()
        pipeline.run_pipeline()
        print("TRAINING PIPELINE EXECUTED SUCCESSFULLY!")
    except Exception as e:
        logging.exception(e)
        raise e
