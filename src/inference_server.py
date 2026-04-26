"""
Reonic Renewable Energy System Designer - FastAPI Inference Server

This server provides a REST API endpoint for predicting optimal renewable energy
system configurations based on customer household features.

Architecture:
    - FastAPI with CORS middleware (allows cross-origin requests from frontend)
    - Loads pre-trained scikit-learn model on startup
    - Inference pipeline: Feature Engineering → ML Model → Heuristic Rules → Cost Estimation
    - Stateless design (no database required)

Endpoints:
    POST /api/design-system: Main inference endpoint
        Input: Customer household features (8 parameters)
        Output: System design recommendation with cost estimate

Usage:
    python src/inference_server.py
    
    Then test with:
    curl -X POST http://localhost:8000/api/design-system \
         -H "Content-Type: application/json" \
         -d '{
             "energy_demand_wh": 6000000,
             "has_ev": 1,
             "has_solar": 0,
             "has_storage": 0,
             "has_wallbox": 1,
             "house_size_sqm": 150,
             "heating_existing_electricity_demand_kwh": 500
         }'
"""

import logging
import math
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from feature_extractor import extract_features

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "model"
MODEL_PATH = MODEL_DIR / "reonic_system_designer.joblib"
SCALER_PATH = MODEL_DIR / "feature_scaler.joblib"
FEATURE_LIST_PATH = MODEL_DIR / "feature_list.txt"
WEB_DIR = PROJECT_ROOT / "web"

# Constants for heuristics and cost estimation
PANEL_CAPACITY_KW = 0.4  # Standard panel capacity (kWp)
PANEL_AREA_SQM = 1.95    # Area per panel (m²)
PANEL_COST_PER_KWP = 1500  # EUR per kWp installed
BATTERY_COST_PER_KWH = 800  # EUR per kWh
EV_ANNUAL_LOAD_KWH = 2500   # Average EV charging load (kWh/year)


# ============================================================================
# Pydantic Request/Response Models
# ============================================================================

class CustomerFeatures(BaseModel):
    """
    Input model for customer household features.

    All numeric fields; boolean flags encoded as 0/1.
    """
    energy_demand_wh: float = Field(
        ...,
        gt=0,
        description="Annual household energy demand in Wh",
        json_schema_extra={"example": 6000000},
    )
    has_ev: int = Field(
        ...,
        ge=0, le=1,
        description="Electric vehicle present (0=no, 1=yes)",
        json_schema_extra={"example": 1},
    )
    has_solar: int = Field(
        ...,
        ge=0, le=1,
        description="Existing solar system (0=no, 1=yes)",
        json_schema_extra={"example": 0},
    )
    has_storage: int = Field(
        ...,
        ge=0, le=1,
        description="Existing battery storage (0=no, 1=yes)",
        json_schema_extra={"example": 0},
    )
    has_wallbox: int = Field(
        ...,
        ge=0, le=1,
        description="Wallbox/EV charger available (0=no, 1=yes)",
        json_schema_extra={"example": 1},
    )
    house_size_sqm: float = Field(
        ...,
        gt=0,
        description="Building area in m²",
        json_schema_extra={"example": 150},
    )
    heating_existing_electricity_demand_kwh: float = Field(
        ...,
        ge=0,
        description="Existing heating load in kWh/year",
        json_schema_extra={"example": 500},
    )


class SystemDesignRecommendation(BaseModel):
    """
    Output model for system design recommendation.
    
    Includes ML-predicted battery size, physics-based solar sizing,
    roof space requirement, and total cost estimate.
    """
    panels_needed: int = Field(
        ...,
        description="Number of 400W solar panels recommended",
        json_schema_extra={"example": 16},
    )
    roof_space_sqm_needed: float = Field(
        ...,
        description="Roof area required for solar panels (m²)",
        json_schema_extra={"example": 31.2},
    )
    recommended_battery_kwh: float = Field(
        ...,
        description="Battery storage capacity recommendation (kWh)",
        json_schema_extra={"example": 8.5},
    )
    estimated_total_cost_euros: float = Field(
        ...,
        description="Total system cost estimate (EUR)",
        json_schema_extra={"example": 19800},
    )


# ============================================================================
# FastAPI Application Setup
# ============================================================================

# Global state: Loaded model and scaler
_model = None
_scaler = None
_features = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load model, scaler, and feature list once at startup; reuse across requests.
    """
    global _model, _scaler, _features

    logger.info("=" * 80)
    logger.info("REONIC SYSTEM DESIGNER - Inference Server Startup")
    logger.info("=" * 80)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}")
    if not FEATURE_LIST_PATH.exists():
        raise FileNotFoundError(f"Feature list not found at {FEATURE_LIST_PATH}")

    logger.info(f"Loading model from {MODEL_PATH}...")
    _model = joblib.load(MODEL_PATH)
    logger.info("✓ Model loaded successfully")

    logger.info(f"Loading scaler from {SCALER_PATH}...")
    _scaler = joblib.load(SCALER_PATH)
    logger.info("✓ Scaler loaded successfully")

    logger.info(f"Loading feature list from {FEATURE_LIST_PATH}...")
    with open(FEATURE_LIST_PATH, "r") as f:
        _features = [line.strip() for line in f if line.strip()]
    logger.info(f"✓ Feature list loaded: {len(_features)} features")
    logger.info(f"  Features: {_features}")

    logger.info("=" * 80)
    logger.info("✓ Server ready for inference")
    logger.info("=" * 80 + "\n")

    yield

    logger.info("Shutting down inference server.")


app = FastAPI(
    title="Reonic Renewable Energy Designer",
    description="ML-powered system design API for residential solar + battery installations",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Middleware: Allow all origins for cross-origin requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Inference Pipeline (shared by all endpoints)
# ============================================================================

def _design_for_features(customer: CustomerFeatures) -> SystemDesignRecommendation:
    """
    Run the 4-step pipeline (feature engineering → ML → physics → cost) for
    one fully-populated CustomerFeatures and return the recommendation.
    """
    logger.info("Step A: Feature Engineering")
    total_estimated_demand_kwh = (
        customer.energy_demand_wh / 1000
        + (EV_ANNUAL_LOAD_KWH if customer.has_ev == 1 else 0)
        + customer.heating_existing_electricity_demand_kwh
    )
    logger.info(f"  Total Estimated Demand: {total_estimated_demand_kwh:.2f} kWh")

    logger.info("Step B: ML Inference (Battery Capacity Prediction)")
    X = pd.DataFrame({
        'total_estimated_demand_kwh': [total_estimated_demand_kwh],
        'energy_demand_wh': [customer.energy_demand_wh],
        'has_ev': [customer.has_ev],
        'has_solar': [customer.has_solar],
        'has_storage': [customer.has_storage],
        'has_wallbox': [customer.has_wallbox],
        'house_size_sqm': [customer.house_size_sqm],
        'heating_existing_electricity_demand_kwh': [customer.heating_existing_electricity_demand_kwh],
    })
    X_scaled = _scaler.transform(X[_features])
    predictions = _model.predict(X_scaled)
    recommended_battery_kwh = round(float(predictions[0, 0]), 1)
    logger.info(f"  Predicted Battery Capacity: {recommended_battery_kwh} kWh")

    logger.info("Step C: Solar Panel Sizing (Physics Heuristic)")
    target_kwp = total_estimated_demand_kwh / 1000
    panels_needed = math.ceil(target_kwp / PANEL_CAPACITY_KW)
    roof_space_sqm_needed = panels_needed * PANEL_AREA_SQM
    logger.info(f"  Panels Needed: {panels_needed}, Roof Space: {roof_space_sqm_needed:.1f} m²")

    logger.info("Step D: Cost Estimation")
    panel_cost = target_kwp * PANEL_COST_PER_KWP
    battery_cost = recommended_battery_kwh * BATTERY_COST_PER_KWH
    estimated_total_cost_euros = panel_cost + battery_cost
    logger.info(f"  Total System Cost: €{estimated_total_cost_euros:,.0f}")

    return SystemDesignRecommendation(
        panels_needed=panels_needed,
        roof_space_sqm_needed=round(roof_space_sqm_needed, 1),
        recommended_battery_kwh=recommended_battery_kwh,
        estimated_total_cost_euros=round(estimated_total_cost_euros, 0),
    )


# ============================================================================
# Inference Endpoint — structured input
# ============================================================================

@app.post(
    "/api/design-system",
    response_model=SystemDesignRecommendation,
    summary="Design optimal renewable energy system",
    description="Predicts battery size using ML model and calculates solar/cost using heuristics",
)
async def design_system(customer: CustomerFeatures) -> SystemDesignRecommendation:
    """Inference endpoint with fully-structured input."""
    logger.info("=" * 80)
    logger.info("Inference Request Received (structured input)")
    logger.info("=" * 80)
    try:
        result = _design_for_features(customer)
        logger.info("=" * 80 + "\n")
        return result
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"System design inference failed: {e}")


# ============================================================================
# Inference Endpoint — free-text prompt (Claude-powered extraction)
# ============================================================================

class PromptRequest(BaseModel):
    """Free-text household description."""

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Plain-language household description, e.g. '150 m² house, EV, 6 MWh/year'.",
        json_schema_extra={"example": "Single family home, ~150 sqm, two EVs, 7 MWh/year, no solar yet."},
    )


class PromptDesignResponse(BaseModel):
    """Recommendation plus the inputs the agent extracted/defaulted."""

    design: SystemDesignRecommendation
    inputs_used: Dict[str, float] = Field(
        ..., description="Final feature values fed to the model (extracted or defaulted).",
    )
    provenance: Dict[str, str] = Field(
        ...,
        description="Per-field origin: 'extracted' (from prompt) or 'defaulted' (training-data median/mode).",
    )


@app.post(
    "/api/design-system/from-prompt",
    response_model=PromptDesignResponse,
    summary="Design a system from a free-text household description",
    description=(
        "Sends the prompt to Claude, which extracts whatever structured fields it can. "
        "Missing fields are filled from training-data defaults. Returns the recommendation "
        "plus a per-field provenance map so the caller can show what was inferred."
    ),
)
async def design_system_from_prompt(req: PromptRequest) -> PromptDesignResponse:
    logger.info("=" * 80)
    logger.info("Inference Request Received (free-text prompt)")
    logger.info(f"  Prompt: {req.prompt!r}")
    logger.info("=" * 80)

    # extract_features() handles Gemini failures internally and falls back to
    # regex — it always returns a fully-populated feature dict.
    features, provenance = extract_features(req.prompt)
    logger.info(f"  Extracted features: {features}")

    try:
        customer = CustomerFeatures(**features)
        design = _design_for_features(customer)
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"System design inference failed: {e}")

    logger.info("=" * 80 + "\n")
    return PromptDesignResponse(design=design, inputs_used=features, provenance=provenance)


# ============================================================================
# Health Check Endpoint
# ============================================================================

@app.get("/health", summary="Health check")
async def health_check() -> Dict[str, str]:
    """
    Simple health check endpoint to verify server is running.
    
    Returns:
        Status message
    """
    return {"status": "healthy", "service": "Reonic System Designer"}


# ============================================================================
# Root + Static UI
# ============================================================================

@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    """Redirect the root to the interactive web UI."""
    return RedirectResponse(url="/web/")


# Mount the single-page UI. `html=True` makes /web/ serve index.html.
if WEB_DIR.exists():
    app.mount("/web", StaticFiles(directory=WEB_DIR, html=True), name="web")
else:
    logger.warning(f"Web UI directory not found at {WEB_DIR}; /web/ disabled.")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    logger.info("Starting Reonic System Designer Inference Server...")
    uvicorn.run(
        "inference_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
