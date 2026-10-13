import io
import os
import sys
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from us_visa.constants import APP_HOST, APP_PORT
from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.pipeline.prediction_pipeline import USvisaClassifier, USvisaData
from us_visa.pipeline.training_pipeline import TrainPipeline

app = FastAPI(
    title="US Visa OFLC Petition Adjudication Risk Engine",
    description="Enterprise Machine Learning REST API for immigration petition risk scoring and adjudication prediction.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PetitionRequest(BaseModel):
    continent: str = Field(..., example="Asia")
    education_of_employee: str = Field(..., example="Master's")
    has_job_experience: str = Field(..., example="Y")
    requires_job_training: str = Field(..., example="N")
    no_of_employees: int = Field(..., example=650)
    yr_of_estab: int = Field(..., example=2010)
    region_of_employment: str = Field(..., example="Northeast")
    prevailing_wage: float = Field(..., example=95000.0)
    unit_of_wage: str = Field(..., example="Year")
    full_time_position: str = Field(..., example="Y")


classifier = USvisaClassifier()


@app.get("/", response_class=HTMLResponse)
async def home():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>US Visa Decision Engine API</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 40px 20px; }
            .container { max-width: 900px; margin: 0 auto; background: #1e293b; border-radius: 16px; padding: 40px; box-shadow: 0 20px 25px -5px rgba(0,0,0,0.5); }
            h1 { color: #38bdf8; margin-top: 0; font-size: 2.2rem; }
            .badge { display: inline-block; background: #0284c7; color: white; padding: 4px 12px; border-radius: 999px; font-size: 0.85rem; font-weight: 600; margin-bottom: 20px; }
            p { color: #94a3b8; line-height: 1.6; font-size: 1.05rem; }
            .endpoints { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 30px 0; }
            .card { background: #0f172a; border: 1px solid #334155; padding: 20px; border-radius: 10px; }
            .card h3 { margin-top: 0; color: #e2e8f0; font-size: 1.1rem; }
            a.btn { display: inline-block; background: #38bdf8; color: #0f172a; text-decoration: none; padding: 10px 20px; border-radius: 8px; font-weight: 700; margin-top: 10px; }
            a.btn:hover { background: #7dd3fc; }
            code { background: #334155; padding: 2px 6px; border-radius: 4px; color: #cbd5e1; }
        </style>
    </head>
    <body>
        <div class="container">
            <span class="badge">Enterprise MLOps REST API</span>
            <h1>US Visa Adjudication Risk Engine</h1>
            <p>High-throughput machine learning inference engine for OFLC / PERM work petition evaluation. Backed by XGBoost ensemble, automated feature engineering, and MongoDB Atlas data synchronization.</p>
            
            <div class="endpoints">
                <div class="card">
                    <h3>📚 Interactive Swagger Docs</h3>
                    <p>Test individual predictions, schemas, and endpoints interactively in your browser.</p>
                    <a href="/docs" class="btn">Open Swagger UI (/docs)</a>
                </div>
                <div class="card">
                    <h3>🚀 Streamlit Visual UI</h3>
                    <p>Launch the rich visual UI with SHAP Explainable AI, Gauge charts, and What-If simulator:</p>
                    <code>streamlit run streamlit_app.py</code>
                </div>
            </div>

            <div class="card">
                <h3>⚡ API Endpoints</h3>
                <ul>
                    <li><code>POST /predict</code> - Single petition risk classification & confidence scoring</li>
                    <li><code>POST /predict_batch</code> - Batch CSV evaluation with aggregate analytics</li>
                    <li><code>GET /train</code> - Trigger asynchronous MLOps model retraining pipeline</li>
                    <li><code>GET /health</code> - Healthcheck & deployed model status</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "US Visa Adjudication Risk Engine",
        "model_file": os.path.exists("saved_models/model.pkl"),
    }


def execute_training_pipeline():
    try:
        logging.info("Asynchronous training pipeline triggered...")
        TrainPipeline().run_pipeline()
    except Exception as e:
        logging.error(f"Background training error: {e}")


@app.get("/train")
async def trigger_training(background_tasks: BackgroundTasks):
    try:
        background_tasks.add_task(execute_training_pipeline)
        return {
            "status": "Training pipeline initiated in background.",
            "pipeline": "Data Ingestion -> Validation -> Transformation -> Trainer -> Evaluation -> Pusher",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
async def predict_petition(request: PetitionRequest):
    try:
        case_data = USvisaData(
            continent=request.continent,
            education_of_employee=request.education_of_employee,
            has_job_experience=request.has_job_experience,
            requires_job_training=request.requires_job_training,
            no_of_employees=request.no_of_employees,
            yr_of_estab=request.yr_of_estab,
            region_of_employment=request.region_of_employment,
            prevailing_wage=request.prevailing_wage,
            unit_of_wage=request.unit_of_wage,
            full_time_position=request.full_time_position,
        )
        df = case_data.get_usvisa_input_data_frame()
        result = classifier.predict_with_details(df)[0]
        return {
            "status": "success",
            "applicant_petition": request.dict(),
            "adjudication_verdict": result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_batch")
async def predict_batch_petitions(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only CSV files are supported.")

        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        preds = classifier.predict_with_details(df)
        pred_df = pd.DataFrame(preds)
        combined_df = pd.concat([df, pred_df], axis=1)

        total = len(combined_df)
        certified = int((combined_df["prediction"] == "Certified").sum())
        denied = int((combined_df["prediction"] == "Denied").sum())

        return {
            "status": "success",
            "file_name": file.filename,
            "total_records": total,
            "certified_count": certified,
            "denied_count": denied,
            "approval_rate_percentage": round((certified / total) * 100, 2) if total > 0 else 0,
            "sample_results": combined_df.head(10).to_dict(orient="records"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=APP_HOST, port=APP_PORT, reload=True)
