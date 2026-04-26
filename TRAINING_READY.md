# Training Dataset Ready! 🎯

## What's Complete

✅ **Phase 1: Target Variables (Y)**  
- Processed `project_options_parts.csv`
- Created `data/project_options_processed.csv` (1,126 projects)
- Three targets: `total_pv_kwp`, `total_inverter_kw`, `total_battery_kwh`

✅ **Phase 2: Feature Variables & Merge**  
- Cleaned `projects_status_quo.csv`  
- Merged with targets on `project_id`
- Created `data/reonic_training_ready.csv` (1,126 samples × 19 columns)

---

## Training Dataset Structure

**File**: `data/reonic_training_ready.csv`

### Columns (19 total)
```
ID Columns (2):
  - project_id
  - option_id

Feature Columns (14):
  - offer_created_at: Timestamp when proposal created
  - first_signed_at: Timestamp when customer signed
  - country: Germany (all samples)
  - energy_demand_wh: Annual household energy demand (Wh)
  - energy_price_per_wh: Local electricity price
  - energy_price_increase: Annual price increase rate
  - load_profile: H0/H1 (household vs small business)
  - has_ev: Boolean (0/1) - EV charger required
  - has_solar: Boolean (0/1) - Existing solar
  - has_storage: Boolean (0/1) - Existing battery
  - has_wallbox: Boolean (0/1) - Wallbox available
  - house_size_sqm: Building area (m²)
  - heating_existing_electricity_demand_kwh: Existing heating load (kWh/year)
  - total_estimated_demand_kwh: DERIVED - Total load = energy + EV (2500 if has_ev) + heating

Target Columns (3):
  - total_pv_kwp: Solar capacity (ground truth)
  - total_inverter_kw: Inverter power (ground truth)
  - total_battery_kwh: Battery storage (ground truth)
```

### Data Quality
- **Samples**: 1,126 projects (complete pairs with both X and Y)
- **Missing values**: Zero in all critical features
- **Data types**: Numeric + categorical (all properly formatted)
- **Duplicates**: None (one row per project_id)

### Target Distribution
```
total_pv_kwp:
  Non-zero: 3 (0.3%) | Range: 0.455-2.375 | Mean: 1.102 (non-zero)
  
total_battery_kwh:
  Non-zero: 1,121 (99.6%) | Range: 3.5-26.98 | Mean: 8.73
  
total_inverter_kw:
  Non-zero: 6 (0.5%) | Range: 5-15 | Mean: 8.33 (non-zero)
```

---

## Data Preprocessing Applied

### 1. Imputation (Domain-Aware)
- `energy_demand_wh`: No missing values (already complete)
- `heating_existing_electricity_demand_kwh`: Filled missing with 0 (assume no heating)
- `house_size_sqm`: Filled missing with median (150 m²)

### 2. Encoding
- Converted boolean columns to integers (0/1): `has_ev`, `has_solar`, `has_storage`, `has_wallbox`

### 3. Feature Engineering
- **`total_estimated_demand_kwh`** = (energy_demand_wh / 1000) + (has_ev × 2500) + heating_electricity_demand_kwh
  - Rationale: Captures total household load for battery sizing
  - EV adds ~2500 kWh/year average charging load

### 4. Columns Dropped (22 non-predictive)
- **Sparse solar/storage details** (>99% missing for non-solar projects): `solar_*`, `storage_*`, `wallbox_charge_speed_kw`
- **Cost/pricing columns** (not relevant for system sizing): `energy_price_*`, `base_price_*`, `ev_annual_drive_distance_km`
- **Metadata** (non-predictive): `request_created_at`, `house_built_year`, `num_inhabitants`
- **Administrative IDs**: `load_profile_editor_id`, `customer_contact_id`

---

## Ready for Model Training

The dataset is now in ideal format for multi-output regression:

1. **Clean & aligned**: One row = one customer with features + optimal system design
2. **No preprocessing needed**: Can feed directly to sklearn model
3. **Balanced by system type**: 
   - 1,121 battery-only projects (primary target)
   - 3 PV-only projects (sparse auxiliary target)
4. **Domain features**: Energy demand, household type, existing systems

---

## Next: Build the ML Model

Use Prompt 3 from `AGENT_PROMPTS.md`:

```
@Renewable Energy ML Engineer

Trigger Skill: Build the Reonic renewable energy ML pipeline.
Task: Write a complete end-to-end Python script that:
[See AGENT_PROMPTS.md - Prompt 3]
```

This will:
1. Load `reonic_training_ready.csv`
2. Scale features with StandardScaler
3. Split train/test (80/20) with stratification
4. Train MultiOutputRegressor with RandomForest
5. Evaluate all three targets
6. Save model + scaler + feature list

---

*Training dataset prepared: 2026-04-26*  
*Ready for model training pipeline*
