# Agent Invocation Prompts

These are ready-to-copy prompts for invoking the **Renewable Energy ML Engineer** agent. Paste directly into Copilot/Cursor chat and tag the agent: `@Renewable Energy ML Engineer`

---

## Prompt 1: Process Target Variables (Y-Variables)

```
Trigger Skill: Prepare the Reonic datasets.
Task: Write a Python script using pandas to process the project_options_parts.csv file. This file represents the "ground truth" designs (our Y-variables).

Specific Instructions:

1. Load the CSV.

2. Filter the dataframe to only include relevant components if necessary (e.g., ignore pure labor costs if they exist, focus on physical hardware).

3. Group the data by project_id and option_id.

4. Aggregate the following custom metrics per option:
   - total_pv_kwp: Sum of (quantity * module_watt_peak / 1000).
   - total_battery_kwh: Sum of battery_capacity_kwh.
   - total_inverter_kw: Sum of inverter_power_kw.

5. Crucial Edge Case: Since a single project_id might have multiple option_ids (e.g., Good, Better, Best proposals), filter the resulting dataframe to only keep the option_id with the maximum total_pv_kwp for each project. We will assume the largest proposed system is the "optimal maximized design" for training purposes.

6. Return a clean dataframe with one row per project_id.

Key Concept Detailed: By forcing the AI to take the maximum PV capacity per project, we solve the problem of duplicate project IDs. We are training your model to predict the "maximum viable system" for a roof. Later, your UI can take that maximum prediction and scale it down to offer the customer cheaper variants.
```

**Status**: ✅ COMPLETE — `data/project_options_processed.csv` ready (1,126 projects × 3 targets)

---

## Prompt 2: Process Feature Variables (X-Variables)

```
Trigger Skill: Prepare the Reonic datasets.
Task: Write a Python script using pandas to process the projects_status_quo.csv file. This file represents household customer features (our X-variables).

Specific Instructions:

1. Load the CSV and profile the data:
   - Identify all feature columns (skip project_id and metadata)
   - Check for missing values and data types
   - Log distributions for numeric features and unique value counts for categorical

2. Data Cleaning:
   - Handle missing values logically:
     * For energy demand: impute with median by region if available, else global median
     * For categorical features (roof_type, etc.): impute with mode or 'unknown'
     * For binary features (EV present, etc.): impute with False (assume no)
   - Validate that no critical features remain empty after imputation

3. Feature Engineering (create derived features):
   - energy_demand_per_sqm = total_energy_demand / house_area (if both available)
   - has_ev = binary flag (1 if EV present, 0 otherwise)
   - Any other domain-relevant ratios or flags that improve predictiveness

4. Categorical Encoding:
   - One-hot encode nominal categorical features (roof_type, building_type, etc.)
   - Keep feature names descriptive (e.g., roof_type_concrete, roof_type_tile)

5. Output:
   - Return a clean dataframe indexed by project_id
   - All rows correspond to projects in project_options_processed.csv
   - No missing values in any feature
   - Save to data/projects_features_processed.csv

Key Concept: The X variables must align perfectly with Y variables by project_id. Later, we will merge X and Y for training. Each row must be a customer household with their measured features (energy demand, house size, EV, etc.).
```

**Status**: ⏳ TODO — Will generate feature matrix aligned with targets

---

## Prompt 3: Build Complete End-to-End Pipeline

```
Trigger Skill: Build the Reonic renewable energy ML pipeline.
Task: Write a complete end-to-end Python script that:

1. Loads and processes both CSVs using the functions from Prompts 1 and 2
2. Merges features (X) and targets (Y) on project_id
3. Handles data imbalance for sparse targets (PV, inverter)
4. Creates train/test split with stratification
5. Trains a multi-output regression model
6. Evaluates all three targets (battery primary, PV/inverter auxiliary)
7. Saves the trained model and artifacts

Specific Requirements:

1. Data Pipeline:
   - Load projects_status_quo.csv → X (features)
   - Load project_options_parts.csv → Y (targets, using Prompt 1 logic)
   - Merge on project_id
   - Check alignment (same number of rows, no NaN after merge)

2. Data Splitting:
   - 80/20 train/test split
   - Stratify on system_type (PV-only, Battery-only, etc.) to balance sparse targets
   - Use random_state=42 for reproducibility

3. Feature Scaling:
   - Use StandardScaler on all numeric features
   - Fit scaler on train data only
   - Apply to both train and test
   - Save scaler to artifacts/

4. Model Architecture:
   - Use MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
   - Alternative: Try GradientBoostingRegressor as ensemble method
   - Apply L1/L2 regularization to handle sparse targets (PV, inverter)

5. Evaluation:
   - Cross-validation: 5-fold CV on training data
   - Report R², MAE, RMSE for each target separately:
     * total_battery_kwh (primary target, abundant data)
     * total_pv_kwp (auxiliary, sparse—3 samples)
     * total_inverter_kw (auxiliary, sparse—6 samples)
   - Plot residuals per target to check for bias

6. Output Artifacts:
   - Save trained model to models/renewable_energy_model.pkl
   - Save scaler to models/scaler.pkl
   - Save feature list to models/feature_names.json
   - Generate evaluation report with train/test metrics
   - Log data shape at each pipeline stage

Philosophy: This is an end-to-end pipeline. The X preprocessing must output data ready for immediate model consumption. No intermediate hand-offs or manual cleanup between phases.
```

**Status**: ⏳ TODO — Will build full training pipeline

---

## Prompt 4: Generate Predictions on New Data

```
Trigger Skill: Score new customer data.
Task: Write a Python script that:

1. Loads the trained model, scaler, and feature list from artifacts/
2. Takes new customer household data (same format as projects_status_quo.csv)
3. Applies the same preprocessing and feature engineering as training
4. Scales features using the saved scaler
5. Generates predictions for solar capacity, battery storage, and inverter power
6. Returns predictions with confidence intervals (uncertainty estimates)

Output format:
   - customer_id, predicted_pv_kwp, predicted_battery_kwh, predicted_inverter_kw
   - Include 95% confidence intervals for each target
```

**Status**: ⏳ TODO — Will create inference pipeline

---

## How to Use

1. **Copy the entire prompt** (including the code block)
2. **Paste into Copilot/Cursor chat**
3. **Tag the agent**: Type `@Renewable Energy ML Engineer` at the start
4. **Hit enter** and let the agent execute the full workflow

Example:
```
@Renewable Energy ML Engineer

Trigger Skill: Prepare the Reonic datasets.
Task: Write a Python script using pandas to process the project_options_parts.csv file...
[paste full prompt]
```

---

*Last Updated: 2026-04-25*
*Part of: Reonic AI Renewable Designer Hackathon*
