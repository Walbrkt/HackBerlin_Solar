# HackBerlin Solar — Reonic Renewable Energy System Designer

End-to-end ML pipeline that recommends a residential solar + battery system from
household features. Customer data → trained `RandomForest` regressor → physics
heuristics for panel/roof sizing → cost estimate, served behind a FastAPI HTTP API.

## Layout

```
CSV/        Raw inputs (projects_status_quo.csv, project_options_parts.csv)
data/       Processed datasets (intermediate + training-ready)
model/      Trained artifacts: reonic_system_designer.joblib, feature_scaler.joblib, feature_list.txt
src/
  process_project_options.py     Phase 1: extract Y targets per project_id
  prepare_training_data.py       Phase 2: clean X features, merge with Y
  train_model.py                 Phase 3: train + evaluate + save artifacts
  inference_server.py            FastAPI server (structured + prompt endpoints)
  feature_extractor.py           Free-text → CustomerFeatures via Gemini (response_schema)
  test_inference_server.py       Client test suite (3 customer profiles)
```

Detailed docs: [PROJECT_COMPLETION.md](PROJECT_COMPLETION.md),
[API_DOCUMENTATION.md](API_DOCUMENTATION.md),
[MODEL_TRAINING_REPORT.md](MODEL_TRAINING_REPORT.md).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run the API

The trained model is committed under [model/](model/), so the server runs
straight after install:

```bash
python src/inference_server.py
# Server: http://localhost:8000
# Swagger UI: http://localhost:8000/docs
```

Smoke-test it from another shell:

```bash
python src/test_inference_server.py
```

Or by hand:

```bash
curl -s -X POST http://localhost:8000/api/design-system \
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
```

Response:

```json
{
  "panels_needed": 23,
  "roof_space_sqm_needed": 44.9,
  "recommended_battery_kwh": 11.6,
  "estimated_total_cost_euros": 22780.0
}
```

## Prompt mode — describe the household in plain English

For users who don't want to fill in every field, `POST /api/design-system/from-prompt`
accepts free text. Gemini (`gemini-2.5-flash`, via Google AI Studio) extracts whichever
fields it can infer; anything missing is filled from training-data medians (4500 kWh
demand, 150 m² house, all booleans 0). Requires `GEMINI_API_KEY` (or `GOOGLE_API_KEY`)
in the server's environment — without it the endpoint returns 503.

```bash
export GEMINI_API_KEY=AIza...   # from https://aistudio.google.com/app/apikey
python src/inference_server.py
```

```bash
curl -s -X POST http://localhost:8000/api/design-system/from-prompt \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Single-family home, ~180 sqm, two EVs charging at home, ~7 MWh per year, no solar yet."}'
```

The response includes the design plus a per-field provenance map so the caller can
show which values were inferred from the prompt vs. assumed:

```json
{
  "design": { "panels_needed": 24, "...": "..." },
  "inputs_used": {
    "energy_demand_wh": 7000000,
    "has_ev": 1,
    "has_wallbox": 1,
    "house_size_sqm": 180,
    "has_solar": 0,
    "has_storage": 0,
    "heating_existing_electricity_demand_kwh": 0
  },
  "provenance": {
    "energy_demand_wh": "extracted",
    "has_ev": "extracted",
    "has_wallbox": "extracted",
    "house_size_sqm": "extracted",
    "has_solar": "defaulted",
    "has_storage": "defaulted",
    "heating_existing_electricity_demand_kwh": "defaulted"
  }
}
```

The extractor uses `client.models.generate_content()` with `response_schema=PartialCustomerFeatures`,
so Gemini returns a Pydantic-validated structured output (no JSON-parsing of free text on our side).

## Retrain from raw CSVs

```bash
python src/process_project_options.py    # CSV/ -> data/project_options_processed.csv
python src/prepare_training_data.py      # -> data/reonic_training_ready.csv
python src/train_model.py                # -> model/*.joblib + feature_list.txt + chart
```

## Inference pipeline

`POST /api/design-system` runs four steps per request:

1. **Feature engineering** — `total_estimated_demand_kwh = energy_demand_wh/1000 + (2500 if has_ev else 0) + heating_existing_electricity_demand_kwh`.
2. **ML inference** — `StandardScaler` → `RandomForestRegressor` → battery kWh (the PV/inverter outputs are sparse in training data and are ignored at serve time; see [MODEL_TRAINING_REPORT.md](MODEL_TRAINING_REPORT.md)).
3. **Physics heuristic** — target solar kWp = total demand / 1000; panels at 0.4 kWp / 1.95 m² each.
4. **Cost** — €1500/kWp PV + €800/kWh battery.

## Notes

- Model R² on battery is low (~4%); demand explains only part of sizing
  variance, so treat output as a baseline and overlay domain rules
  (budget, incentives, available SKUs) downstream. See
  [MODEL_TRAINING_REPORT.md](MODEL_TRAINING_REPORT.md) for full evaluation.
- CORS is wide open (`allow_origins=["*"]`) for hackathon demo use; lock it
  down before any non-local deployment.
