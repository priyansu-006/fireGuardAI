"""FastAPI Web Application and REST API for Forest Fire Classification.

Provides:
- Web UI Dashboard at http://localhost:8000/
- Prediction REST API at POST /api/predict
- HTML Form Submission at POST /predict and GET /?Temperature=...
- Model Benchmark Metrics at GET /api/metrics
- Static file serving for plots and UI assets
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.config import BEST_MODEL_PATH, MODEL_COMPARISON_CSV, PLOTS_DIR
from src.predict import ForestFirePredictor

app = FastAPI(
    title="Forest Fire ML Prediction API & Dashboard",
    description="Real-time Wildfire Risk Classification using Canadian Fire Weather Indices",
    version="1.0.0",
)

# Initialize Predictor
predictor: Optional[ForestFirePredictor] = None
try:
    if BEST_MODEL_PATH.exists():
        predictor = ForestFirePredictor(BEST_MODEL_PATH)
        print(f"[Backend] Predictor loaded successfully from {BEST_MODEL_PATH}")
    else:
        print(f"[Backend Warning] Model file not found at {BEST_MODEL_PATH}. Run python -m src.train first.")
except Exception as e:
    print(f"[Backend Error] Failed to initialize predictor: {e}")

# Static directories
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# Mount outputs/plots to serve generated figures
if PLOTS_DIR.exists():
    app.mount("/plots", StaticFiles(directory=str(PLOTS_DIR)), name="plots")

# Mount static web assets
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class FirePredictionRequest(BaseModel):
    """Schema for individual inference requests."""
    Temperature: float = Field(..., description="Noon Temperature in Celsius (e.g. 32.0)", ge=10.0, le=60.0)
    RH: float = Field(..., description="Relative Humidity % (e.g. 55.0)", ge=0.0, le=100.0)
    Ws: float = Field(..., description="Wind Speed in km/h (e.g. 15.0)", ge=0.0, le=60.0)
    Rain: float = Field(..., description="Daily Rainfall in mm (e.g. 0.0)", ge=0.0, le=200.0)
    FFMC: float = Field(..., description="Fine Fuel Moisture Code (e.g. 85.0)", ge=0.0, le=110.0)
    DMC: float = Field(..., description="Duff Moisture Code (e.g. 15.0)", ge=0.0, le=150.0)
    DC: float = Field(..., description="Drought Code (e.g. 50.0)", ge=0.0, le=500.0)
    ISI: float = Field(..., description="Initial Spread Index (e.g. 6.0)", ge=0.0, le=60.0)
    BUI: float = Field(..., description="Buildup Index (e.g. 18.0)", ge=0.0, le=150.0)
    FWI: float = Field(..., description="Fire Weather Index (e.g. 10.0)", ge=0.0, le=80.0)
    Region: str = Field(default="Bejaia", description="Region name: 'Bejaia' or 'Sidi-Bel Abbes'")


def run_inference_internal(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Helper to run inference through predictor."""
    global predictor
    if predictor is None:
        if BEST_MODEL_PATH.exists():
            predictor = ForestFirePredictor(BEST_MODEL_PATH)
        else:
            raise RuntimeError("Model pipeline not found. Train the model first.")
    result = predictor.predict(input_dict)[0]
    return result


@app.get("/")
def serve_dashboard(
    Temperature: Optional[float] = None,
    RH: Optional[float] = None,
    Ws: Optional[float] = None,
    Rain: Optional[float] = None,
    FFMC: Optional[float] = None,
    DMC: Optional[float] = None,
    DC: Optional[float] = None,
    ISI: Optional[float] = None,
    BUI: Optional[float] = None,
    FWI: Optional[float] = None,
    Region: Optional[str] = "Bejaia",
):
    """Serve the dashboard, and handle query-param fallback if form was submitted via GET."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h1>Dashboard loading... please wait.</h1>")

    # If query parameters were passed from a standard browser GET form submit
    if Temperature is not None and RH is not None and FFMC is not None:
        try:
            payload = {
                "Temperature": float(Temperature),
                "RH": float(RH),
                "Ws": float(Ws or 15.0),
                "Rain": float(Rain or 0.0),
                "FFMC": float(FFMC),
                "DMC": float(DMC or 15.0),
                "DC": float(DC or 50.0),
                "ISI": float(ISI or 6.0),
                "BUI": float(BUI or 18.0),
                "FWI": float(FWI or 10.0),
                "Region": str(Region or "Bejaia"),
            }
            res = run_inference_internal(payload)
            # Read index.html and inject precalculated values
            with open(index_file, "r", encoding="utf-8") as f:
                content = f.read()
            # Inject preloaded script
            injection = f"""
            <script>
            window.addEventListener('DOMContentLoaded', () => {{
                if (typeof renderPredictionResult === 'function') {{
                    renderPredictionResult({res}, {payload});
                }}
            }});
            </script>
            """
            content = content.replace("</body>", f"{injection}</body>")
            return HTMLResponse(content)
        except Exception as e:
            print(f"[Backend Query Error] {e}")

    return FileResponse(index_file)


@app.post("/api/predict")
def predict_fire_risk_json(req: FirePredictionRequest) -> Dict[str, Any]:
    """Execute real-time wildfire classification via JSON API."""
    try:
        input_dict = req.model_dump()
        result = run_inference_internal(input_dict)
        return {
            "status": "success",
            "prediction": result,
            "inputs": input_dict,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict")
async def predict_fire_risk_form(
    Temperature: float = Form(...),
    RH: float = Form(...),
    Ws: float = Form(...),
    Rain: float = Form(...),
    FFMC: float = Form(...),
    DMC: float = Form(...),
    DC: float = Form(...),
    ISI: float = Form(...),
    BUI: float = Form(...),
    FWI: float = Form(...),
    Region: str = Form("Bejaia"),
):
    """Handle standard HTML Form POST submissions."""
    payload = {
        "Temperature": Temperature,
        "RH": RH,
        "Ws": Ws,
        "Rain": Rain,
        "FFMC": FFMC,
        "DMC": DMC,
        "DC": DC,
        "ISI": ISI,
        "BUI": BUI,
        "FWI": FWI,
        "Region": Region,
    }
    try:
        res = run_inference_internal(payload)
        return JSONResponse({"status": "success", "prediction": res, "inputs": payload})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/metrics")
def get_model_metrics():
    """Retrieve the model comparison benchmark results table."""
    if not MODEL_COMPARISON_CSV.exists():
        raise HTTPException(status_code=404, detail="Model comparison metrics CSV not found.")
    df = pd.read_csv(MODEL_COMPARISON_CSV)
    return {
        "status": "success",
        "models": df.to_dict(orient="records"),
    }


@app.get("/api/plots")
def list_available_plots():
    """List all available generated evaluation and EDA plots."""
    if not PLOTS_DIR.exists():
        return {"plots": []}
    plots = [
        {
            "name": p.stem.replace("_", " ").title(),
            "filename": p.name,
            "url": f"/plots/{p.name}",
        }
        for p in PLOTS_DIR.glob("*.png")
    ]
    return {"status": "success", "plots": plots}


@app.get("/api/india/states")
def get_india_states():
    """Retrieve all 35 Indian States & UTs with historical wildfire incidents and vulnerability rankings."""
    try:
        from src.india_analytics import get_all_states_ranking
        states = get_all_states_ranking()
        return {"status": "success", "total_states": len(states), "states": states}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/india/summary")
def get_india_national_summary():
    """Retrieve national aggregate wildfire statistics across all Indian States & UTs."""
    try:
        from src.india_analytics import get_national_summary_stats
        summary = get_national_summary_stats()
        return {"status": "success", "summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/india/state/{state_name}")
def get_india_state_detail(state_name: str):
    """Retrieve detailed historical wildfire records and forest belt profile for a specific state."""
    try:
        from src.india_analytics import get_state_profile
        profile = get_state_profile(state_name)
        return {"status": "success", "profile": profile}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 70)
    print("Launching Forest Fire ML Dashboard on http://localhost:8000")
    print("Press Ctrl+C to terminate.")
    print("=" * 70 + "\n")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
