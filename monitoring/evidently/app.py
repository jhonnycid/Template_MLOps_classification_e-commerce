from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import json
import os
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from datetime import datetime

app = FastAPI()

# Créer un dossier templates si nécessaire
os.makedirs("templates", exist_ok=True)

# Créer un fichier de template HTML de base
with open("templates/index.html", "w") as f:
    f.write("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Evidently Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #333; }
            .card { border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin-bottom: 20px; }
            .success { color: green; }
            .error { color: red; }
            pre { background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto; }
        </style>
    </head>
    <body>
        <h1>Evidently Dashboard</h1>
        
        <div class="card">
            <h2>Status</h2>
            <p>Reference data: <span class="{{ 'success' if reference_data_set else 'error' }}">{{ 'Set' if reference_data_set else 'Not set' }}</span></p>
            <p>Last updated: {{ last_updated }}</p>
        </div>
        
        <div class="card">
            <h2>Drift Information</h2>
            {% if drift_data %}
                <p>Drift detected: <span class="{{ 'error' if drift_data.drift_detected else 'success' }}">{{ drift_data.drift_detected }}</span></p>
                <p>Drift score: {{ drift_data.drift_score }}</p>
                <h3>Detailed Report</h3>
                <pre>{{ drift_data.report|tojson(indent=2) }}</pre>
            {% else %}
                <p>No drift data available. Please send data to /drift/detect endpoint.</p>
            {% endif %}
        </div>
        
        <div class="card">
            <h2>API Endpoints</h2>
            <ul>
                <li><code>POST /drift/reference</code> - Set reference data</li>
                <li><code>POST /drift/detect</code> - Detect drift</li>
                <li><code>GET /health</code> - Health check</li>
            </ul>
        </div>
    </body>
    </html>
    """)

# Configurer les templates
templates = Jinja2Templates(directory="templates")

# Stocker les données de référence
reference_data = None
last_updated = "Never"
drift_data = None

class DataInput(BaseModel):
    data: List[Dict[str, Any]]
    reference: Optional[bool] = False

@app.get("/", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    return templates.TemplateResponse(
        "index.html", 
        {
            "request": request, 
            "reference_data_set": reference_data is not None,
            "last_updated": last_updated,
            "drift_data": drift_data
        }
    )

@app.post("/drift/reference")
async def set_reference_data(input_data: DataInput):
    global reference_data, last_updated
    try:
        reference_data = pd.DataFrame(input_data.data)
        last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return {"status": "success", "message": "Reference data set successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error setting reference data: {str(e)}")

@app.post("/drift/detect")
async def detect_drift(input_data: DataInput, background_tasks: BackgroundTasks):
    global reference_data, drift_data
    if reference_data is None:
        raise HTTPException(status_code=400, detail="Reference data not set")
    
    try:
        current_data = pd.DataFrame(input_data.data)

        # Générer le rapport Evidently avec la nouvelle API
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference_data, current_data=current_data)

        json_report = report.json()
        report_data = json.loads(json_report)

        # Extraire le score de dérive global
        drift_result = report_data.get("metrics", [])[0]
        drift_detected = drift_result["result"]["dataset_drift"]
        drift_score = drift_result["result"]["number_of_drifted_columns"] / drift_result["result"]["number_of_columns"]

        # Stocker les résultats pour l'affichage dans l'interface utilisateur
        drift_data = {
            "drift_detected": drift_detected,
            "drift_score": drift_score,
            "report": report_data
        }

        if input_data.reference:
            background_tasks.add_task(update_reference_data, current_data)

        return drift_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detecting drift: {str(e)}")

def update_reference_data(new_reference):
    global reference_data
    reference_data = new_reference

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8050)
