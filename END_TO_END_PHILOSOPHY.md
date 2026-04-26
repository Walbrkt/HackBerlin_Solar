# End-to-End Pipeline Philosophy

## Why Option 3: End-to-End (not Options 1 or 2)

The Reonic renewable energy ML pipeline requires an **end-to-end lifecycle perspective**, not isolated phases:

### ❌ Why NOT Option 1 (EDA + Feature Engineering Only)
- Data exploration and feature engineering are only **half** the battle
- Without thinking about downstream model requirements, preprocessing can create friction
- Missing the "integration point" between clean data and model consumption
- Result: Clean data but unprepared for the ML model

### ❌ Why NOT Option 2 (Model Development Only)
- Multi-output regression is powerful but **useless without proper preprocessing**
- Model development assumes clean, aligned (X, Y) pairs—preprocessing must deliver exactly that
- Skipping upstream work leads to model failures during training
- Result: Model code but incompatible data inputs

### ✅ Why Option 3 (End-to-End Pipeline)
- Keeps entire lifecycle in the agent's context window
- When writing data cleaning, the agent **proactively formats for downstream model**
- Preprocessing functions output exactly what the regression model expects
- No silos; each phase feeds naturally into the next
- Result: Single modular pipeline from raw CSV → trained model

### 🔧 Reinforced by Option 4 (Code Generation & Debugging)
- Ensures agent **actually writes ML code**, not just theoretical advice
- Proactive debugging and error fixing during execution
- Moves from "Here's how you might..." to "Here's the complete working pipeline"
- Result: Production-ready code, not documentation

---

## Pipeline Architecture (as Designed)

```
Raw CSVs
   ↓
[EDA] Profile data, identify issues
   ↓
[Features] Clean X variables, align to project_id
[Targets] Aggregate Y variables by project_id
   ↓
[Merge] Join X and Y on project_id, ensure alignment
   ↓
[Split] Train/test split with stratification
   ↓
[Model] Multi-output regression (battery primary, PV/inverter auxiliary)
   ↓
[Evaluate] Metrics, cross-validation, residual analysis
   ↓
Trained Model + Artifacts (joblib, scalers, feature lists)
```

**Key insight**: Step 3 (Merge) is the critical bridge. Preprocessing must output data ready for model consumption without additional cleanup.

---

## Workflow Expectations

When invoked with a task like **"Build a complete ML pipeline from both CSVs"**:

1. Agent loads both CSVs and profiles them (EDA)
2. Agent writes preprocessing functions that output clean, aligned (X, Y)
3. Agent constructs train/test split accounting for data imbalance (sparse PV/inverter targets)
4. Agent selects and trains multi-output model
5. Agent evaluates all three targets and reports metrics
6. Agent delivers: single modular script + trained model + validation report

**No intermediate hand-offs. No manual data cleaning between phases.**

---

## Technical Guardrails

The agent enforces:
- **Reproducibility**: Logging at each pipeline stage shows data shape and transformations
- **Domain logic**: Handles sparse targets (PV, inverter) with appropriate regularization
- **Code quality**: Modular functions, type hints, docstrings, error handling
- **Downstream compatibility**: Output format matches scikit-learn MultiOutputRegressor expectations

---

*Strategy document for Reonic hackathon ML pipeline*
